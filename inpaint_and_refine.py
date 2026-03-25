import random
import argparse
from pytorch_lightning import seed_everything
import os
import subprocess
import ffmpeg
from torchvision import transforms
import torch
import numpy as np
import einops
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from m2svid.utils.video_utils import open_ffmpeg_process, get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation

parser = argparse.ArgumentParser()
parser.add_argument( "--model_config", type=str)
parser.add_argument( "--ckpt", type=str)
parser.add_argument( "--video_path", type=str)
parser.add_argument( "--grid_video_path", type=str, help="Grid video where left half is mask and right half is wrapped video")
parser.add_argument( "--output_folder", type=str)
parser.add_argument( "--reprojected_closing_holes_kernel", type=int, default=11)
parser.add_argument( "--mask_antialias", type=int, default=False)
parser.add_argument( "--spatial_tile_size", type=int, default=512)
parser.add_argument( "--spatial_tile_overlap", type=int, default=256)
parser.add_argument( "--temporal_tile_size", type=int, default=14)
parser.add_argument( "--temporal_tile_overlap", type=int, default=4)
args = parser.parse_args()


seed = random.randint(0, 65535)
seed_everything(seed)

config = OmegaConf.load(args.model_config)
denoising_model = instantiate_from_config(config.model).cpu()
denoising_model.init_from_ckpt(args.ckpt)
denoising_model = denoising_model.half().eval()

reprojected_closing_holes_kernel = args.reprojected_closing_holes_kernel
mask_antialias = args.mask_antialias
output_folder = args.output_folder

# load and preprocess videos
input_video = get_video_frames(args.video_path, normalize=False)
grid_video = get_video_frames(args.grid_video_path, normalize=False)
W_half = grid_video.shape[3] // 2
reprojected_mask = grid_video[:, 0:1, :, :W_half]
reprojected = grid_video[:, :, :, W_half:]
fps = get_video_fps(args.video_path, ffmpeg.probe(args.video_path))

reprojected_mask = apply_closing(reprojected_mask, reprojected_closing_holes_kernel)
reprojected[reprojected_mask.repeat(1, 3, 1, 1) > 0.5] = 0
reprojected_mask = apply_dilation(reprojected_mask, 3)
reprojected_mask = reprojected_mask.repeat(1, 3, 1, 1)

import torch.nn.functional as F

input_video = input_video.permute(1, 0, 2, 3) # [c,t,h,w], uint8 0-255
reprojected = reprojected.permute(1, 0, 2, 3) # [c,t,h,w], uint8 0-255
reprojected_mask = reprojected_mask.permute(1, 0, 2, 3) # [c,t,h,w], uint8 0/1

# Match dimensions to prevent tensor size mismatch during generation
H_iv, W_iv = input_video.shape[2:]
H_rp, W_rp = reprojected.shape[2:]

if H_iv != H_rp or W_iv != W_rp:
    target_H = max(H_iv, H_rp)
    target_W = max(W_iv, W_rp)
    
    if H_iv < target_H or W_iv < target_W:
        # Pad with 0 (black in uint8 space)
        input_video = F.pad(input_video, (0, target_W - W_iv, 0, target_H - H_iv), value=0)
    if H_rp < target_H or W_rp < target_W:
        reprojected = F.pad(reprojected, (0, target_W - W_rp, 0, target_H - H_rp), value=0)
        reprojected_mask = F.pad(reprojected_mask, (0, target_W - W_rp, 0, target_H - H_rp), value=0)

c, T, H, W = reprojected_mask.shape
downsampled_resolution = [int(H / 8), int(W / 8)]
reprojected_mask = reprojected_mask.permute(1, 0, 2, 3).float() # [t,c,h,w]
reprojected_mask = transforms.Resize(downsampled_resolution, antialias=mask_antialias)(reprojected_mask)
reprojected_mask = reprojected_mask[:, [0]]
reprojected_mask = reprojected_mask.permute(1, 0, 2, 3).half() * 2.0 - 1.0 # [c,t,h,w]

latent_H, latent_W = H // 8, W // 8
final_latents = torch.zeros((1, 4, T, latent_H, latent_W), dtype=torch.float16, device="cpu")
final_weights = torch.zeros((1, 1, T, latent_H, latent_W), dtype=torch.float16, device="cpu")

max_temporal_size = getattr(denoising_model, "num_samples", args.temporal_tile_size)
if args.temporal_tile_size > max_temporal_size:
    print(f"Warning: temporal_tile_size ({args.temporal_tile_size}) exceeds model's max num_samples ({max_temporal_size}). Capping to {max_temporal_size}.")
    args.temporal_tile_size = max_temporal_size

t_stride = max(1, args.temporal_tile_size - args.temporal_tile_overlap)
h_stride = max(1, args.spatial_tile_size - args.spatial_tile_overlap)
w_stride = max(1, args.spatial_tile_size - args.spatial_tile_overlap)

t_size = args.temporal_tile_size
h_size = args.spatial_tile_size
w_size = args.spatial_tile_size

# Tiling bounds
def get_bounds(length, size, stride):
    bounds = []
    for start in range(0, length, stride):
        end = min(start + size, length)
        if end - start < size and length >= size:
            start = end - size
        bounds.append((start, end))
        if end == length:
            break
    return bounds

t_bounds = get_bounds(T, t_size, t_stride)
h_bounds = get_bounds(H, h_size, h_stride)
w_bounds = get_bounds(W, w_size, w_stride)

print(f"Total tiles to process: {len(t_bounds)} temporal x {len(h_bounds)} spatial height x {len(w_bounds)} spatial width")

video_name = os.path.splitext(os.path.basename(args.video_path))[0]
os.makedirs(output_folder, exist_ok=True)
out_path = os.path.join(output_folder, f'{video_name}_generated.mp4')
print(f"Streaming generated chunks into {out_path} as they complete...")

ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-loglevel', 'error',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{W}x{H}',
    '-pix_fmt', 'rgb24',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-x264opts', 'rc-lookahead=10',
    '-crf', '14',
    '-profile:v', 'high10',
    '-pix_fmt', 'yuv420p10le',
    out_path
]

process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
first_stage_model = denoising_model.first_stage_model
last_decoded_frame = 0

from tqdm import tqdm

for i, (t_s, t_e) in enumerate(tqdm(t_bounds, desc="Temporal Chunks")):
    spatial_pbar = tqdm(total=len(h_bounds) * len(w_bounds), desc=f"Spatial Tiles for frames {t_s}-{t_e}", leave=False)
    for h_s, h_e in h_bounds:
        for w_s, w_e in w_bounds:
            # slice arrays
            iv_slice = input_video[:, t_s:t_e, h_s:h_e, w_s:w_e]
            rp_slice = reprojected[:, t_s:t_e, h_s:h_e, w_s:w_e]
            # masks are already downsampled to latent size
            lh_s, lh_e = h_s // 8, h_e // 8
            lw_s, lw_e = w_s // 8, w_e // 8
            rm_slice = reprojected_mask[:, t_s:t_e, lh_s:lh_e, lw_s:lw_e]
            
            # Map uint8 [0, 255] to float16 [-1, 1] strictly on the GPU chunk
            iv_slice_gpu = (iv_slice.cuda().half() / 255.0) * 2.0 - 1.0
            rp_slice_gpu = (rp_slice.cuda().half() / 255.0) * 2.0 - 1.0

            # send to GPU
            input_batch = {
                'video': iv_slice_gpu[None],
                'video_2nd_view': iv_slice_gpu[None],
                'reprojected_video': rp_slice_gpu[None],
                'reprojected_mask': rm_slice[None].cuda(), # already downsampled and in [-1, 1]
                'fps_id': torch.tensor([fps]).cuda(),
                'caption': [""],
                "motion_bucket_id": torch.tensor([127]).cuda()
            }
            
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16):
                    out = denoising_model.generate(input_batch, do_not_decode=True)
                    generated_latent_flat = out['generated-video']
                    
            # Flat generated-video is (B*T, C, h_latent, w_latent). Rearrange it to B C T h w
            generated_latent = einops.rearrange(generated_latent_flat, '(b t) c h w -> b c t h w', b=1, t=t_e-t_s).cpu()
            
            final_latents[:, :, t_s:t_e, lh_s:lh_e, lw_s:lw_e] += generated_latent
            final_weights[:, :, t_s:t_e, lh_s:lh_e, lw_s:lw_e] += 1.0
            
            # Memory management (let PyTorch handle cache efficiently)
            del out
            del input_batch
            del generated_latent_flat
            del generated_latent
            
            spatial_pbar.update(1)
            
    spatial_pbar.close()

    # Determine which frames are fully resolved and decode them to ffmpeg stream
    if i < len(t_bounds) - 1:
        next_t_s = t_bounds[i+1][0]
    else:
        next_t_s = T
        
    if next_t_s > last_decoded_frame:
        # We can decode from last_decoded_frame to next_t_s - 1
        resolved_latents = final_latents[:, :, last_decoded_frame:next_t_s, :, :] / final_weights[:, :, last_decoded_frame:next_t_s, :, :]
        
        # Bring VAE to GPU for these frames
        first_stage_model.to('cuda')
        
        for f_idx in range(next_t_s - last_decoded_frame):
            frame_latent = resolved_latents[:, :, f_idx:f_idx+1, :, :].cuda()  # (1, 4, 1, H/8, W/8)
            frame_latent_flat = einops.rearrange(frame_latent, 'b c t h w -> (b t) c h w')
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16):
                    decoded = denoising_model.decode_first_stage(frame_latent_flat, num_video_frames=1) # (1, 3, H, W)
                    
            frame_numpy = decoded[0].detach().cpu().float().numpy().transpose(1, 2, 0) # H, W, 3
            frame_numpy = (((frame_numpy + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
            process.stdin.write(frame_numpy.tobytes())
            process.stdin.flush()
            
            # Clear tensor
            del decoded, frame_latent, frame_latent_flat
            
        first_stage_model.to('cpu')
        torch.cuda.empty_cache()
        
        # Free up CPU RAM for the latents we just wrote by zeroing it out
        final_latents[:, :, last_decoded_frame:next_t_s, :, :] = 0
        
        last_decoded_frame = next_t_s

process.stdin.close()
process.wait()
print("Done processing!")
