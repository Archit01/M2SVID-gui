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

import argparse
from m2svid.utils.video_utils import open_ffmpeg_process, read_frames_in_batches_ffmpeg, get_video_fps
from m2svid.warping.warping import scatter_image

import numpy as np
import tqdm
from pathlib import Path
import ffmpeg
import os
import cv2
from depth_preprocess import preprocess_depth_frame


def process_video_with_depth(
    video_path,
    depth_path,
    output_path,
    disparity_scale=None,
    disparity_perc=None,
    batch_size=10,
    global_normalize=False,
    start_frame=0,
    max_frames=None,
    crf=14,
    bit_depth=10,
    dilate_x=0.0,
    dilate_y=0.0,
    blur_x=0,
    blur_y=0,
    dilate_left=0.0,
    blur_left=0,
    blur_left_mix=0.5,
    use_cuda=False,
    micro_hole_strength=0,
):
    # Probe input video
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = get_video_fps(video_path, probe)

    total_video_frames = None
    if 'nb_frames' in video_stream:
        total_video_frames = int(video_stream['nb_frames'])

    # Validate Depth Path type (video or numpy)
    is_depth_video = depth_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    min_depth = None
    max_depth = None
    relative_depth = None
    depth_w = None
    depth_h = None
    total_depth_frames = 0

    print("Analyzing depth map for global min/max estimation...")
    if is_depth_video:
        depth_probe = ffmpeg.probe(depth_path)
        depth_stream = next(s for s in depth_probe['streams'] if s['codec_type'] == 'video')
        depth_w = int(depth_stream['width'])
        depth_h = int(depth_stream['height'])
        if 'nb_frames' in depth_stream:
            total_depth_frames = int(depth_stream['nb_frames'])
        else:
            total_depth_frames = total_video_frames if total_video_frames else 999999

        if global_normalize:
            # First pass: read all depth frames to find global min & max
            current_min = float('inf')
            current_max = float('-inf')
            for depth_frames in tqdm.tqdm(
                read_frames_in_batches_ffmpeg(depth_path, batch_size, depth_w, depth_h, start_sec=0),
                desc="Finding global min/max depth"
            ):
                # Using the red channel (assuming identical RGB for grayscale depth)
                d_batch = depth_frames[..., 0].astype(np.float32)
                current_min = min(current_min, d_batch.min())
                current_max = max(current_max, d_batch.max())
            min_depth, max_depth = current_min, current_max
            print(f"Global depth min: {min_depth}, max: {max_depth}")
    else:
        # Load NumPy array
        relative_depth_data = np.load(depth_path)
        relative_depth = relative_depth_data['depth'] if 'depth' in relative_depth_data else relative_depth_data
        total_depth_frames = relative_depth.shape[0]
        depth_h, depth_w = relative_depth.shape[1], relative_depth.shape[2]
        if global_normalize:
            min_depth, max_depth = relative_depth.min(), relative_depth.max()
            print(f"Global depth min: {min_depth}, max: {max_depth}")


    # Limit frames if max_frames is provided
    end_frame = start_frame + max_frames if max_frames is not None else total_depth_frames
    frames_to_process = min(end_frame - start_frame, total_depth_frames - start_frame)
    start_sec = start_frame / fps if fps else 0.0

    mode_str = "CUDA (GPU)" if use_cuda else "NumPy (CPU)"
    print(f"Processing frames {start_frame} to {start_frame + frames_to_process} using {mode_str}...")

    if disparity_perc is not None:
        current_disparity_scale = int(width * disparity_perc)
    else:
        current_disparity_scale = disparity_scale

    ffmpeg_process_grid = None

    video_frame_generator = read_frames_in_batches_ffmpeg(video_path, batch_size, width, height, start_sec=start_sec)
    
    if is_depth_video:
        depth_frame_generator = read_frames_in_batches_ffmpeg(depth_path, batch_size, depth_w, depth_h, start_sec=start_sec)
    
    frames_processed = 0

    for i in tqdm.tqdm(range(0, frames_to_process, batch_size), desc="Warping implementation"):
        try:
            left_frames = next(video_frame_generator)
            if is_depth_video:
                depth_frames_rgb = next(depth_frame_generator)
                depth_batch = depth_frames_rgb[..., 0].astype(np.float32) # take red channel
            else:
                depth_batch = relative_depth[start_frame + frames_processed : start_frame + frames_processed + len(left_frames)]
                depth_batch = depth_batch.astype(np.float32)
        except StopIteration:
            break

        # Ensure same size limits
        current_batch_size = min(len(left_frames), len(depth_batch))
        if current_batch_size == 0:
            break
            
        left_frames = left_frames[:current_batch_size]
        depth_batch = depth_batch[:current_batch_size]
        
        # Apply Global Normalization if required
        if global_normalize and min_depth is not None and max_depth is not None:
            if max_depth > min_depth:
                depth_batch = (depth_batch - min_depth) / (max_depth - min_depth)
        elif is_depth_video:
            # Fallback to map 0-255 MP4 data into 0-1 range to match .npz scales
            depth_batch = depth_batch / 255.0

        # Space alignment & Scaling
        depth_batch_resized = np.array([
            cv2.resize(d_frame, (width, height), interpolation=cv2.INTER_CUBIC)
            for d_frame in depth_batch
        ])

        # Apply depth preprocessing if any parameters are set
        for fi in range(len(depth_batch_resized)):
            depth_batch_resized[fi] = preprocess_depth_frame(
                depth_batch_resized[fi],
                dilate_x=dilate_x,
                dilate_y=dilate_y,
                blur_x=blur_x,
                blur_y=blur_y,
                dilate_left=dilate_left,
                blur_left=blur_left,
                blur_left_mix=blur_left_mix,
                use_cuda=use_cuda,
            )

        disparities = depth_batch_resized * current_disparity_scale

        reprojected_right_videos = []
        reprojected_right_masks = []

        for left_frame, disparity in zip(left_frames, disparities):
            reprojected_image, inpainting_mask, _ = scatter_image(
                left_frame, disparity, direction=-1, scale_factor=1, reproject_depth=True, use_cuda=use_cuda,
                close_micro_holes=(micro_hole_strength > 0), micro_hole_iters=micro_hole_strength
            )
            reprojected_right_videos.append(reprojected_image)
            reprojected_right_masks.append(inpainting_mask)

        reprojected_right_videos = np.stack(reprojected_right_videos, axis=0)
        reprojected_right_masks = np.stack(reprojected_right_masks, axis=0)

        # Initialize writers natively supporting custom encoding 
        if ffmpeg_process_grid is None:
            ffmpeg_process_grid = open_ffmpeg_process(
                output_path, width * 2, height, fps, crf=crf, bit_depth=bit_depth
            )

        for reprojected_frame, mask_frame in zip(
            reprojected_right_videos, reprojected_right_masks
        ):
            if len(mask_frame.shape) == 2:
                mask_rgb = np.stack((mask_frame,)*3, axis=-1)
            elif mask_frame.shape[-1] == 1:
                mask_rgb = np.concatenate((mask_frame,)*3, axis=-1)
            else:
                mask_rgb = mask_frame
            grid_frame = np.concatenate((mask_rgb, reprojected_frame), axis=1)
            # Ensure the array is robust for the FFmpeg pipe (uint8 and contiguous in memory)
            grid_frame = np.ascontiguousarray(grid_frame.astype(np.uint8))
            
            # Use chunked writes to prevent OSError 22 on Windows for large buffers
            frame_bytes = grid_frame.tobytes()
            chunk_size = 1024 * 1024 # 1MB chunks
            
            try:
                for j in range(0, len(frame_bytes), chunk_size):
                    ffmpeg_process_grid.stdin.write(frame_bytes[j : j + chunk_size])
                ffmpeg_process_grid.stdin.flush()
            except (OSError, BrokenPipeError) as e:
                # Capture stderr if available to diagnose the real cause of the crash
                error_log = ""
                try:
                    if ffmpeg_process_grid.stderr:
                        # Non-blocking read attempt
                        import fcntl
                        import os
                        # On Windows, fcntl doesn't exist. We'll use a safer approach.
                    if os.name != 'nt':
                        import fcntl
                        import os
                        fd = ffmpeg_process_grid.stderr.fileno()
                        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    
                    error_log = ffmpeg_process_grid.stderr.read().decode('utf-8', errors='ignore')
                except:
                    error_log = "Could not capture FFmpeg stderr."
                
                print(f"\n[FFmpeg Error] Pipe write failed: {e}")
                if error_log:
                    print(f"--- FFmpeg Log ---\n{error_log}\n------------------")
                raise
            
        frames_processed += current_batch_size

    if ffmpeg_process_grid:
        ffmpeg_process_grid.stdin.close()
        ffmpeg_process_grid.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with depth data to generate reprojected videos and masks.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to the depth file (.npz, .npy, or .mp4/video).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output grid video.")
    parser.add_argument("--disparity_scale", type=float, default=None, help="Absolute disparity scale to apply.")
    parser.add_argument("--disparity_perc", type=float, default=None, help="Percentage based disparity scale to apply.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for chunk processing.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process in this chunk.")
    parser.add_argument("--global_normalize", action="store_true", help="Apply strict global normalization for the entire depth video/map across all frames.")
    parser.add_argument("--crf", type=int, default=14, help="CRF value for the H264 10-bit output.")
    parser.add_argument("--bit_depth", type=int, default=10, help="Bit depth for the H264 output.")
    parser.add_argument("--batch_size", type=int, default=10, help="Processing sliding window / chunk size in frames.")
    # Depth preprocessing arguments
    parser.add_argument("--dilate_x", type=float, default=0.0, help="Depth dilation X (negative = erosion).")
    parser.add_argument("--dilate_y", type=float, default=0.0, help="Depth dilation Y (negative = erosion).")
    parser.add_argument("--blur_x", type=int, default=0, help="Depth blur kernel X.")
    parser.add_argument("--blur_y", type=int, default=0, help="Depth blur kernel Y.")
    parser.add_argument("--dilate_left", type=float, default=0.0, help="Directional left dilation.")
    parser.add_argument("--blur_left", type=int, default=0, help="Edge-aware blur left.")
    parser.add_argument("--blur_left_mix", type=float, default=0.5, help="H/V balance for blur_left (0=H, 1=V).")
    parser.add_argument("--use_cuda", action="store_true", help="Use CuPy GPU-accelerated warping (falls back to NumPy if unavailable).")
    parser.add_argument("--micro_hole_strength", type=float, default=0.0, help="Strength of morphological closing for small holes (0=off, 1-5=iterations, supports fractions like 0.5).")

    args = parser.parse_args()

    assert (args.disparity_scale is None) or (args.disparity_perc is None)
    assert (args.disparity_scale is not None) or (args.disparity_perc is not None)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    process_video_with_depth(
        args.video_path,
        args.depth_path,
        args.output_path,
        disparity_scale=args.disparity_scale,
        disparity_perc=args.disparity_perc,
        batch_size=args.batch_size,
        global_normalize=args.global_normalize,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        crf=args.crf,
        bit_depth=args.bit_depth,
        dilate_x=args.dilate_x,
        dilate_y=args.dilate_y,
        blur_x=args.blur_x,
        blur_y=args.blur_y,
        dilate_left=args.dilate_left,
        blur_left=args.blur_left,
        blur_left_mix=args.blur_left_mix,
        use_cuda=args.use_cuda,
        micro_hole_strength=args.micro_hole_strength,
    )
