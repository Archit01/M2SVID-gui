"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch

from bidavideo.models.bidastereo_model import BiDAStereoModel
from m2svid.warping.warping import scatter_image, scatter_image_gpu, _TORCH_CUDA_AVAILABLE
from m2svid.utils.video_utils import open_ffmpeg_process, read_frames_in_batches_ffmpeg, get_video_fps, get_total_frames, split_left_right, get_video_frames, \
        save_disparity_as_png


### Needed to be loaded first
model = BiDAStereoModel()
ckpt = 'checkpoints/bidastereo_robust/bidastereo_robust.pth'
strict = True
state_dict = torch.load(ckpt)
if "model" in state_dict:
    state_dict = state_dict["model"]
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
model.model.load_state_dict(state_dict, strict=strict)
print("Done loading model checkpoint", ckpt)

DEVICE = 'cuda'
model.to(DEVICE)
model.eval()

import json
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import ffmpeg
import os
import tqdm
import traceback
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict

from utils.loftr import compute_feature_matching_loftr


def process_video(name, REWRITE):
    video_path = f'{video_root}/{name}'
    output_path_cropped_video  = f'{output_root_cropped_videos}/{name}'
    output_path_crop_params  = f'{output_root_crop_params}/{name}'
    output_path_reprojected = f'{output_root_reprojected}/{name}'
    output_path_mask = f'{output_root_mask}/{name}'
    output_path_disparities_info = f'{output_root_disparities_info}/{name}'
    output_path_disparity = f'{output_root_disparity}/{name}'
    debug_output_path = f'{debug_root}/{name}' if debug_root is not None else None
    os.makedirs(output_path_disparity, exist_ok=True)
    print(video_path)

    ffmpeg_process_reprojected  = None
    ffmpeg_process_mask  = None

    if os.path.exists(output_path_disparities_info + '.npz'):
        print(f'Video: {name} is already processed.')
        if not REWRITE:
            return
        else:
            print(f'Rewriting: {name} .')
    else:
        print(f'Processing {name}.')

    if os.path.exists(output_path_reprojected):
        os.remove(output_path_reprojected)
    if os.path.exists(output_path_mask):
        os.remove(output_path_mask)
    if os.path.exists(output_path_cropped_video):
        os.remove(output_path_cropped_video)

    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = get_video_fps(video_path, probe)
        total_frames = get_total_frames(video_path, probe)
        max_disparities = []
        min_disparities = []
        mean_disparities = []
        mse_errors = []

        # ------------------------------------------------------------------------------------------------------------------------------
        # Apply shift
        import kornia.feature as KF

        duration = total_frames / fps
        if args.max_fps is None:
            shift_fps = nframes_for_shift / duration
        else:
            shift_fps = min(args.max_fps, nframes_for_shift / duration)

        frames = get_video_frames(video_path, fps=shift_fps, num_frames=nframes_for_shift)
        frames = frames.transpose(0, 3, 1, 2)
        frames = torch.from_numpy(frames)
        print(f"Estimating shift by {len(frames)} frames")
        left_videos, right_videos = split_left_right(frames, rectified=True)

        all_mkpts0, all_mkpts1, Fm, min_hor_disp, robust_min_hor_disp, max_abs_ver_disp, robust_max_abs_ver_disp \
                = compute_feature_matching_loftr(matcher_indoor, left_videos, right_videos, debug_output=debug_output_path,
                                    draw_matches=True, cutoff_min_disparity=robust_threshold)
        print('indoor', min_hor_disp, robust_min_hor_disp)

        all_mkpts0_o, all_mkpts1_o, Fm_o, min_hor_disp_o, robust_min_hor_disp_o, max_abs_ver_disp_o, robust_max_abs_ver_disp_o = \
            compute_feature_matching_loftr(matcher_outdoor, left_videos, right_videos, debug_output=debug_output_path,
                                    draw_matches=True, cutoff_min_disparity=robust_threshold)

        print('outdoor', min_hor_disp_o, robust_min_hor_disp_o)
        if len(all_mkpts0_o) > len(all_mkpts0):
            print(f'outdoor has more matched points {len(all_mkpts0_o)} > {len(all_mkpts0)}, using outdoor')
            all_mkpts0, all_mkpts1, Fm, min_hor_disp, robust_min_hor_disp, max_abs_ver_disp, robust_max_abs_ver_disp  = \
                all_mkpts0_o, all_mkpts1_o, Fm_o, min_hor_disp_o, robust_min_hor_disp_o, max_abs_ver_disp_o, robust_max_abs_ver_disp_o
        else:
            print(f'indoor has more matched points {len(all_mkpts0)} > {len(all_mkpts0_o)}, using indoor')

        robust_shift = max(0, -robust_min_hor_disp + target_min_disparity)
        robust_shift = int(np.ceil(robust_shift / 4)) * 4 # that after shift the width will be divisable on 2 (ffmpeg error)

        usual_shift = max(0, -min_hor_disp + target_min_disparity)
        usual_shift = int(np.ceil(usual_shift / 4)) * 4 # that after shift the width will be divisable on 2 (ffmpeg error)

        shift = robust_shift
        print("FINAL shift", shift)

        with open(output_path_crop_params + '.json', 'w') as fout:
            json.dump({
                'robust_shift': robust_shift,
                'shift': usual_shift,
                'min_hor_disp': min_hor_disp,
                'robust_min_hor_disp': robust_min_hor_disp,
                'max_abs_ver_disp': max_abs_ver_disp,
                'robust_max_abs_ver_disp': robust_max_abs_ver_disp
                }, fout)

        # ------------------------------------------------------------------------------------------------------------------------------

        for i, frames in enumerate(tqdm.tqdm(read_frames_in_batches_ffmpeg(video_path, batch_size, width, height), total=int(total_frames // batch_size))):

            frames = frames.transpose(0, 3, 1, 2)
            frames = torch.from_numpy(frames)

            left_videos, right_videos = split_left_right(frames, rectified=True)

            if len(left_videos) == 1:
                break

            left_videos = left_videos[:,:,:, :-int(shift // 2)]
            right_videos = right_videos[:,:,:, int(shift // 2):]
            _, _, original_h, original_w = left_videos.shape

            batch_dict = defaultdict(list)
            if DOWNSCALE is not None:
                transform = transforms.Resize([int(original_h // DOWNSCALE), int(original_w // DOWNSCALE)], interpolation=InterpolationMode.BICUBIC, antialias=True)
                input_left_videos = transform(left_videos)
                input_right_videos = transform(right_videos)
            else:
                input_left_videos = left_videos
                input_right_videos = right_videos

            batch_dict["stereo_video"] = torch.stack([input_left_videos / 255., input_right_videos / 255.], dim=1)

            with torch.no_grad():
                predictions = model(batch_dict)
                disparities = predictions['raw_disparity']
                disparities = - disparities

                if DOWNSCALE is not None:
                    disparities = disparities * DOWNSCALE

                # saving
                if DOWNSCALE is not None:
                    downscaled_disparities = disparities
                else:
                    transform = transforms.Resize([int(original_h // 2), int(original_w // 2)], interpolation=InterpolationMode.BICUBIC, antialias=True)
                    downscaled_disparities = transform(disparities)

                for j, disparity in enumerate(downscaled_disparities):
                    assert disparity.shape[0] == 1
                    disparity = disparity[0]
                    disparity = disparity.cpu().numpy()
                    output_path = f'{output_path_disparity}/{i * batch_size + j}.png'
                    save_disparity_as_png(disparity, output_path)
                    # np.savez_compressed(f'{output_path_disparity}_{i}', array=disparities)

                if DOWNSCALE is not None:
                    _, _, original_h, original_w = left_videos.shape
                    transform = transforms.Resize([original_h, original_w], interpolation=InterpolationMode.BICUBIC, antialias=False)
                    disparities = transform(disparities)

            assert disparities.shape[1] == 1
            disparities = disparities[:, 0]

            left_videos = left_videos.numpy().transpose(0, 2, 3, 1)
            right_videos = right_videos.numpy().transpose(0, 2, 3, 1)
            disparities = disparities.numpy()
            cropped_video = np.concatenate((left_videos, right_videos), axis=2)

            reprojected_right_videos = []
            reprojected_right_masks = []

            _scatter_fn = scatter_image_gpu if _TORCH_CUDA_AVAILABLE else scatter_image
            for left_frame, right_frame, disparity in zip(left_videos, right_videos, disparities):
                reprojected_image, inpainting_mask, reprojected_depth = _scatter_fn(left_frame, disparity, direction=-1, scale_factor=1, reproject_depth=True)
                reprojected_right_videos.append(reprojected_image)
                reprojected_right_masks.append(inpainting_mask)

                error = (np.abs(reprojected_image - right_frame) ** 2).sum(axis=-1)
                error = error.flatten()[inpainting_mask.flatten() == 0]
                mse_errors.append(error.mean())


            reprojected_right_videos = np.stack(reprojected_right_videos, axis=0)
            reprojected_right_masks = np.stack(reprojected_right_masks, axis=0)

            max_disparities.extend(disparities.max(axis=-1).max(axis=-1))
            min_disparities.extend(disparities.min(axis=-1).min(axis=-1))
            mean_disparities.extend(disparities.mean(axis=-1).mean(axis=-1))

            if ffmpeg_process_reprojected is None:
                _, height, width, _ = reprojected_right_videos.shape
                ffmpeg_process_reprojected = open_ffmpeg_process(output_path_reprojected, width, height, fps)
                ffmpeg_process_mask = open_ffmpeg_process(output_path_mask, width, height, fps, grayscale=True, no_compression=True)
                ffmpeg_process_cropped_video = open_ffmpeg_process(output_path_cropped_video, width * 2, height, fps)

            ffmpeg_process_reprojected.stdin.write(np.ascontiguousarray(reprojected_right_videos.astype(np.uint8)).tobytes())
            ffmpeg_process_mask.stdin.write(np.ascontiguousarray(reprojected_right_masks.astype(np.uint8)).tobytes())
            ffmpeg_process_cropped_video.stdin.write(np.ascontiguousarray(cropped_video.astype(np.uint8)).tobytes())

            del frames, left_videos, right_videos, reprojected_right_videos, reprojected_right_masks

            if args.debug:
                break

        ffmpeg_process_reprojected.stdin.close()
        ffmpeg_process_mask.stdin.close()

        ffmpeg_process_reprojected.wait()
        ffmpeg_process_mask.wait()

        ffmpeg_process_cropped_video.stdin.close()
        ffmpeg_process_cropped_video.wait()

        min_disparities = np.array(min_disparities)
        max_disparities = np.array(max_disparities)
        mse_errors = np.array(mse_errors)

        # Save compressed
        np.savez_compressed(output_path_disparities_info, max_disparities=max_disparities,
                    min_disparities=min_disparities, mean_disparities=mean_disparities, mse_errors=mse_errors)

        print(f'Processing {name} is done.')

    except Exception:
        print(f'--------------- Error processing video: {name} --------------------')
        traceback.print_exc()
    finally:
        if ffmpeg_process_reprojected is not None:
            ffmpeg_process_reprojected.stdin.close()
            ffmpeg_process_mask.stdin.close()

            ffmpeg_process_reprojected.wait()
            ffmpeg_process_mask.wait()

            ffmpeg_process_cropped_video.stdin.close()
            ffmpeg_process_cropped_video.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Divide data into n_parts and select a specific part.")
    parser.add_argument("--part", type=int, default=0, help="The part index to select (0-based).")
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--target_min_disparity", type=int, default=1)
    parser.add_argument("--robust_threshold", default=0.001, type=float)
    parser.add_argument("--max_fps", default=4, type=float)
    parser.add_argument("--debug", default=False, action='store_true')
    args = parser.parse_args()

    REWRITE = False
    nframes_for_shift = 200
    batch_size = args.batch_size
    robust_threshold = args.robust_threshold
    DOWNSCALE = 2
    target_min_disparity = args.target_min_disparity
    LOFTR_SHIFT = True

    with open('datasets/ego4d/subsets/all_full_videos.json') as fin:
        data = json.load(fin)
    split_data = np.array_split(data, args.n_parts)
    data = split_data[args.part].tolist()
    print(f"Processing part {args.part} out of {args.n_parts}: {len(data)} videos")


    dataset_root = 'datasets/ego4d'
    video_root = f'{dataset_root}/clips/rectified_videos'
    output_root_cropped_videos = f'{dataset_root}/clips/cropped_videos'
    output_root_crop_params = f'{dataset_root}/clips/crop_params'
    output_root_reprojected = f'{dataset_root}/clips/reprojected'
    output_root_mask = f'{dataset_root}/clips/reprojected_mask'
    output_root_disparities_info = f'{dataset_root}/clips/disparity_info'
    output_root_disparity = f'{dataset_root}/clips/disparity'
    for folder in [output_root_cropped_videos, output_root_crop_params, output_root_reprojected,
                    output_root_mask, output_root_disparities_info, output_root_disparity]:
        os.makedirs(folder, exist_ok=True)

    if args.debug:
        debug_root = f'{dataset_root}/examples/run_stereo_and_reproject/clips'
        os.makedirs(debug_root, exist_ok=True)
    else:
        debug_root = None

    data = ['04ec9004-7f3a-4d48-b48e-fc4eb36d74ac_clip_0000.mp4']

    import kornia.feature as KF
    matcher_outdoor = KF.LoFTR(pretrained="outdoor").cuda()
    matcher_indoor = KF.LoFTR(pretrained="indoor_new").cuda()

    for name in data:
        process_video(name, REWRITE)
