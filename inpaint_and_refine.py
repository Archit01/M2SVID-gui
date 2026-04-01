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
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from m2svid.utils.video_utils import open_ffmpeg_process, get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation

parser = argparse.ArgumentParser()
parser.add_argument("--model_config", type=str)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--video_path", type=str)
parser.add_argument("--grid_video_path", type=str, help="Grid video where left half is mask and right half is wrapped video")
parser.add_argument("--output_folder", type=str)
parser.add_argument("--reprojected_closing_holes_kernel", type=int, default=11)
parser.add_argument("--mask_antialias", type=int, default=False)
parser.add_argument("--spatial_tile_size", type=int, default=512)
parser.add_argument("--spatial_tile_overlap", type=int, default=256)
# New temporal chunking arguments
parser.add_argument("--chunk_size", type=int, default=25, help="Total frames per model forward pass")
parser.add_argument("--overlap", type=int, default=3, help="Number of frames to overlap and cross-fade")
parser.add_argument("--original_input_blend_strength", type=float, default=0.0, help="Weight of original input for conditioning")
parser.add_argument("--dry_run", action="store_true", help="Print chunk schedule and exit without running the model")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Chunk schedule builder
# ---------------------------------------------------------------------------
def build_chunk_schedule(total_frames, chunk_size, overlap):
    """Build a list of chunks describing which source frames to use and how to handle overlaps.
    
    Each entry is a dict:
        source_indices: list[int] - contiguous indices into the source video
        actual_len: int - number of valid frames generated before padding
        overlap: int - frames overlapping with previous chunk (to be cross-faded)
        tail_overlap: int - frames overlapping with next chunk (to be cached)
        abs_start: int - the absolute starting frame index
        padded: int - how many pad frames were appended
    """
    if total_frames <= 0:
        return []

    stride = max(1, chunk_size - overlap)
    schedule = []
    
    for i in range(0, total_frames, stride):
        end_idx = min(i + chunk_size, total_frames)
        actual_len = end_idx - i
        
        # Skip useless tail chunks that contribute almost no new frames
        if i > 0 and overlap > 0 and actual_len <= overlap:
            break
            
        src_indices = list(range(i, end_idx))
        padded = 0
        while len(src_indices) < chunk_size:
            src_indices.append(src_indices[-1])
            padded += 1
            
        is_last = (i + stride >= total_frames) or (total_frames - (i + stride) <= overlap)
        
        schedule.append({
            'source_indices': src_indices,
            'actual_len': actual_len,
            'padded': padded,
            'overlap': overlap if i > 0 else 0,
            'tail_overlap': overlap if not is_last else 0,
            'abs_start': i,
        })
        
    return schedule


# ---------------------------------------------------------------------------
# Spatial tiling helpers (unchanged from original)
# ---------------------------------------------------------------------------
def get_spatial_bounds(length, size, stride):
    bounds = []
    for start in range(0, length, stride):
        end = min(start + size, length)
        if end - start < size and length >= size:
            start = end - size
        bounds.append((start, end))
        if end == length:
            break
    return bounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
seed = random.randint(0, 65535)
seed_everything(seed)

config = OmegaConf.load(args.model_config)
denoising_model = instantiate_from_config(config.model).cpu()
denoising_model.init_from_ckpt(args.ckpt)
denoising_model = denoising_model.half().eval()

# Clamp chunk_size to model maximum
max_temporal_size = getattr(denoising_model, "num_samples", args.chunk_size)
if args.chunk_size > max_temporal_size:
    print(f"Warning: chunk_size ({args.chunk_size}) exceeds model's max num_samples ({max_temporal_size}). Capping to {max_temporal_size}.")
    args.chunk_size = max_temporal_size

reprojected_closing_holes_kernel = args.reprojected_closing_holes_kernel
mask_antialias = args.mask_antialias
output_folder = args.output_folder

# Load and preprocess videos
input_video = get_video_frames(args.video_path, normalize=False)
grid_video = get_video_frames(args.grid_video_path, normalize=False)
W_half = grid_video.shape[3] // 2
reprojected_mask = grid_video[:, 0:1, :, :W_half]
reprojected = grid_video[:, :, :, W_half:]
fps = get_video_fps(args.video_path, ffmpeg.probe(args.video_path))

reprojected_mask = apply_closing(reprojected_mask, reprojected_closing_holes_kernel)
reprojected[reprojected_mask.repeat(1, 3, 1, 1) > 0.5] = 0
reprojected_mask = apply_dilation(reprojected_mask, 3)
# reprojected_mask = reprojected_mask.repeat(1, 3, 1, 1)  # Keep as 1-channel to save RAM

input_video = input_video.permute(1, 0, 2, 3)       # [c,t,h,w], uint8 0-255
reprojected = reprojected.permute(1, 0, 2, 3)       # [c,t,h,w], uint8 0-255
reprojected_mask = reprojected_mask.permute(1, 0, 2, 3)  # [c,t,h,w], uint8 0/1

# Match dimensions - Grid resolution is authoritative
H_iv, W_iv = input_video.shape[2:]
H_rp, W_rp = reprojected.shape[2:]

# Ensure Grid resolution is a multiple of 8 to avoid VAE size mismatch (e.g. 802 -> 800)
target_H = (H_rp // 8) * 8
target_W = (W_rp // 8) * 8

# Resize reprojected and mask if they aren't multiples of 8 (required for VAE consistency)
if H_rp != target_H or W_rp != target_W:
    print(f"Warning: Grid resolution ({W_rp}x{H_rp}) is not a multiple of 8. Resizing to {target_W}x{target_H} to avoid model mismatch.")
    reprojected = F.interpolate(reprojected.float(), size=(target_H, target_W), mode='bicubic', align_corners=False).to(reprojected.dtype).clamp(0, 255)
    reprojected_mask = F.interpolate(reprojected_mask.float(), size=(target_H, target_W), mode='bicubic', align_corners=False).to(reprojected_mask.dtype).clamp(-1, 1)

# Resize input_video to match target resolution exactly (no crops, irrespective of aspect ratio)
if H_iv != target_H or W_iv != target_W:
    input_video = F.interpolate(input_video.float(), size=(target_H, target_W), mode='bicubic', align_corners=False).to(input_video.dtype).clamp(0, 255)

c, T, H, W = reprojected_mask.shape
downsampled_resolution = [int(H / 8), int(W / 8)]

# Perform resizing in chunks to avoid large float32 allocations (Peak RAM reduction)
print(f"Resizing mask chunks for latent space ({T} frames)...")
resized_masks = []
chunk_size_resize = 100 
for i in range(0, T, chunk_size_resize):
    # Convert chunk to float for resizing, but keep channel count at 1
    m_chunk = reprojected_mask[:, i:i+chunk_size_resize].permute(1, 0, 2, 3).float()
    # Normalize/clamp to [0, 1] before resizing if needed, but here they are 0/1
    m_chunk = transforms.Resize(
        downsampled_resolution, 
        interpolation=transforms.InterpolationMode.BICUBIC, 
        antialias=mask_antialias
    )(m_chunk).clamp(0, 1)
    resized_masks.append(m_chunk.half()) # Store as half precision [0, 1]

reprojected_mask = torch.cat(resized_masks, dim=0) # [T, 1, LH, LW]
reprojected_mask = reprojected_mask.permute(1, 0, 2, 3) # [1, T, LH, LW]
reprojected_mask = reprojected_mask * 2.0 - 1.0  # Scale to [-1, 1] for model input

latent_H, latent_W = H // 8, W // 8

# Spatial tiling setup
h_stride = max(1, args.spatial_tile_size - args.spatial_tile_overlap)
w_stride = max(1, args.spatial_tile_size - args.spatial_tile_overlap)
h_size = args.spatial_tile_size
w_size = args.spatial_tile_size
h_bounds = get_spatial_bounds(H, h_size, h_stride)
w_bounds = get_spatial_bounds(W, w_size, w_stride)

# Build chunk schedule
chunk_schedule = build_chunk_schedule(T, args.chunk_size, args.overlap)

print(f"\n=== Chunk Schedule (total_frames={T}, chunk_size={args.chunk_size}, overlap={args.overlap}) ===")
for ci, ch in enumerate(chunk_schedule):
    src = ch['source_indices']
    print(f"  Chunk {ci}: source[{src[0]}..{src[-1]}] "
          f"(pad={ch['padded']}) -> output [{ch['abs_start']}..{ch['abs_start'] + ch['actual_len'] - 1}] "
          f"({ch['actual_len']} active frames) "
          f"[overlap={ch['overlap']}, tail_overlap={ch['tail_overlap']}]")

total_output = sum(ch['actual_len'] for ch in chunk_schedule) - sum(ch['overlap'] for ch in chunk_schedule)
print(f"  Total output frames: {total_output} (expected: {T})")
assert total_output == T, f"Output frame count mismatch: {total_output} != {T}"

if args.dry_run:
    print("\n--dry_run: exiting without running the model.")
    exit(0)

print(f"\nSpatial tiles: {len(h_bounds)} height x {len(w_bounds)} width")

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

overlap_buffer = []  # To store decoded pixel frames from the tail of the previous chunk

for ci, chunk_info in enumerate(tqdm(chunk_schedule, desc="Temporal Chunks")):
    src_indices = chunk_info['source_indices']
    c_overlap = chunk_info['overlap']
    c_tail_overlap = chunk_info['tail_overlap']
    abs_start = chunk_info['abs_start']
    actual_len = chunk_info['actual_len']
    n_gen = len(src_indices)  # == chunk_size (after padding)

    # Gather source frames in reordered order  [c, chunk_size, h, w]
    iv_chunk = torch.stack([input_video[:, idx] for idx in src_indices], dim=1)
    rp_chunk = torch.stack([reprojected[:, idx] for idx in src_indices], dim=1)
    
    # Optional original input conditioning blend for overlap region
    if ci > 0 and c_overlap > 0 and overlap_buffer and args.original_input_blend_strength > 0:
        blend_strength = args.original_input_blend_strength
        for f_rel in range(min(c_overlap, len(overlap_buffer))):
            w = (f_rel / (c_overlap - 1)) if c_overlap > 1 else 0.5
            w = w * blend_strength
            orig_frame_tensor = rp_chunk[:, f_rel]
            prev_gen_np = overlap_buffer[f_rel]
            prev_gen_tensor = torch.tensor(prev_gen_np).permute(2, 0, 1).float().to(orig_frame_tensor.device)
            rp_chunk[:, f_rel] = ((1.0 - w) * prev_gen_tensor + w * orig_frame_tensor).to(rp_chunk.dtype)

    # Mask is in latent space [c, t, lh, lw] — gather similarly
    rm_chunk = torch.stack([reprojected_mask[:, idx] for idx in src_indices], dim=1)

    # Allocate per-chunk latent accumulator (for spatial blending)
    chunk_latent_H, chunk_latent_W = latent_H, latent_W
    chunk_latents = torch.zeros((1, 4, n_gen, chunk_latent_H, chunk_latent_W), dtype=torch.float16, device="cpu")
    chunk_weights = torch.zeros((1, 1, n_gen, chunk_latent_H, chunk_latent_W), dtype=torch.float16, device="cpu")

    spatial_pbar = tqdm(total=len(h_bounds) * len(w_bounds),
                        desc=f"Spatial Tiles for chunk {ci} (frames {abs_start}-{abs_start+actual_len-1})",
                        leave=False)

    for h_s, h_e in h_bounds:
        for w_s, w_e in w_bounds:
            # Slice spatial region
            iv_slice = iv_chunk[:, :, h_s:h_e, w_s:w_e]
            rp_slice = rp_chunk[:, :, h_s:h_e, w_s:w_e]
            lh_s, lh_e = h_s // 8, h_e // 8
            lw_s, lw_e = w_s // 8, w_e // 8
            rm_slice = rm_chunk[:, :, lh_s:lh_e, lw_s:lw_e]

            # Map uint8 [0,255] → float16 [-1,1]
            iv_slice_gpu = (iv_slice.cuda().half() / 255.0) * 2.0 - 1.0
            rp_slice_gpu = (rp_slice.cuda().half() / 255.0) * 2.0 - 1.0

            input_batch = {
                'video': iv_slice_gpu[None],
                'video_2nd_view': iv_slice_gpu[None],
                'reprojected_video': rp_slice_gpu[None],
                'reprojected_mask': rm_slice[None].cuda(),
                'fps_id': torch.tensor([fps]).cuda(),
                'caption': [""],
                "motion_bucket_id": torch.tensor([127]).cuda()
            }

            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16):
                    out = denoising_model.generate(input_batch, do_not_decode=True)
                    generated_latent_flat = out['generated-video']

            generated_latent = einops.rearrange(
                generated_latent_flat, '(b t) c h w -> b c t h w', b=1, t=n_gen
            ).cpu()

            # Spatial blending weight
            weight = torch.ones((1, 1, n_gen, lh_e - lh_s, lw_e - lw_s), dtype=torch.float16, device="cpu")
            latent_h_ovr = args.spatial_tile_overlap // 8
            latent_w_ovr = args.spatial_tile_overlap // 8
            if h_s > 0:
                ramp = torch.linspace(0, 1, latent_h_ovr, dtype=torch.float16).view(1, 1, 1, -1, 1)
                weight[:, :, :, :latent_h_ovr, :] *= ramp
            if h_e < H:
                ramp = torch.linspace(1, 0, latent_h_ovr, dtype=torch.float16).view(1, 1, 1, -1, 1)
                weight[:, :, :, -latent_h_ovr:, :] *= ramp
            if w_s > 0:
                ramp = torch.linspace(0, 1, latent_w_ovr, dtype=torch.float16).view(1, 1, 1, 1, -1)
                weight[:, :, :, :, :latent_w_ovr] *= ramp
            if w_e < W:
                ramp = torch.linspace(1, 0, latent_w_ovr, dtype=torch.float16).view(1, 1, 1, 1, -1)
                weight[:, :, :, :, -latent_w_ovr:] *= ramp

            chunk_latents[:, :, :, lh_s:lh_e, lw_s:lw_e] += generated_latent * weight
            chunk_weights[:, :, :, lh_s:lh_e, lw_s:lw_e] += weight

            del out, input_batch, generated_latent_flat, generated_latent
            spatial_pbar.update(1)

    spatial_pbar.close()

    # Resolve spatial blending
    resolved = chunk_latents / chunk_weights  # (1, 4, n_gen, lH, lW)
    del chunk_latents, chunk_weights

    # Decode kept frames and stream to ffmpeg
    first_stage_model.to('cuda')
    new_overlap_buffer = []

    for f_rel in range(0, actual_len):
        frame_latent = resolved[:, :, f_rel:f_rel + 1, :, :].cuda()
        frame_latent_flat = einops.rearrange(frame_latent, 'b c t h w -> (b t) c h w')
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16):
                decoded = denoising_model.decode_first_stage(frame_latent_flat, num_video_frames=1)

        frame_numpy = decoded[0].detach().cpu().float().numpy().transpose(1, 2, 0)
        frame_numpy_uint8 = (((frame_numpy + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
        
        abs_idx = abs_start + f_rel
        final_frame = frame_numpy_uint8
        
        # Crossfade overlapping head frames with previous tail
        if ci > 0 and f_rel < c_overlap and f_rel < len(overlap_buffer):
            w = f_rel / (c_overlap - 1) if c_overlap > 1 else 0.5
            prev_frame = overlap_buffer[f_rel]
            blended = (1.0 - w) * prev_frame.astype(np.float32) + w * final_frame.astype(np.float32)
            final_frame = blended.clip(0, 255).astype(np.uint8)

        # Output to ffmpeg unless this is a tail frame of a non-last chunk
        is_tail = (f_rel >= actual_len - c_tail_overlap)
        if not is_tail:
            process.stdin.write(np.ascontiguousarray(final_frame).tobytes())
        else:
            new_overlap_buffer.append(final_frame)

        # Feed this generated frame back as conditioning for future chunks
        if abs_idx < T:
            reprojected[:, abs_idx] = torch.tensor(final_frame).permute(2, 0, 1)
            reprojected_mask[:, abs_idx] = -1.0  # mark as unmasked

        del decoded, frame_latent, frame_latent_flat

    first_stage_model.to('cpu')
    torch.cuda.empty_cache()
    process.stdin.flush()
    del resolved
    overlap_buffer = new_overlap_buffer

process.stdin.close()
process.wait()
print("Done processing!")
