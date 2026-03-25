import os
import glob
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from decord import VideoReader, cpu
import logging
import time
import re
import argparse
import sys
import threading
from tqdm import tqdm
import gc

from core.common.video_io import start_ffmpeg_pipe_process, get_video_stream_info
from core.common.gpu_utils import release_cuda_memory
from core.common.sidecar_manager import SidecarConfigManager, find_video_by_core_name, find_sidecar_file, read_clip_sidecar
from core.common.image_processing import (
    apply_mask_dilation, apply_gaussian_blur, apply_shadow_blur,
    apply_color_transfer, apply_borders_to_frames,
    apply_dubois_anaglyph_torch, apply_optimized_anaglyph_torch
)
from core.common.file_organizer import move_files_to_finished

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _read_ffmpeg_output(pipe, log_level):
    try:
        for line in iter(pipe.readline, b""):
            if line:
                pass  # suppress ffmpeg logs from polluting tqdm
    except Exception as e:
        logger.error(f"Error reading FFmpeg pipe: {e}")
    finally:
        if pipe:
            pipe.close()


def run_batch_process(settings, single_video_path=None):
    if settings is None:
        return

    if single_video_path and os.path.exists(single_video_path):
        inpainted_videos = [single_video_path]
        single_mode = True
    else:
        inpainted_videos = sorted(glob.glob(os.path.join(settings["inpainted_folder"], "*.mp4")))
        single_mode = False

    if not inpainted_videos:
        logger.info("No .mp4 files found in the inpainted video folder.")
        return

    resume_enabled = settings.get("resume", False)
    if resume_enabled:
        finished_dir = os.path.join(settings["inpainted_folder"], "finished")
        if os.path.isdir(finished_dir):
            finished_files = set(os.listdir(finished_dir))
            original_count = len(inpainted_videos)
            inpainted_videos = [v for v in inpainted_videos if os.path.basename(v) not in finished_files]
            skipped_count = original_count - len(inpainted_videos)
            if skipped_count > 0:
                logger.info(f"Resume: Skipped {skipped_count} already processed files.")
        else:
            logger.info("Resume: No 'finished' folder found. Processing all files.")

    if not inpainted_videos:
        logger.info("All videos have already been processed (Resume mode).")
        return

    total_videos = len(inpainted_videos)
    sidecar_manager = SidecarConfigManager()

    for i, inpainted_video_path in enumerate(inpainted_videos):
        if single_mode and i > 0:
            break

        base_name = os.path.basename(inpainted_video_path)
        
        # Apply per-video overrides if available
        per_video_overrides = settings.get("per_video_overrides", {})
        active_settings = dict(settings)  # copy global settings
        if base_name in per_video_overrides:
            overrides = per_video_overrides[base_name]
            active_settings.update(overrides)
            logger.info(f"Loaded per-video settings for {base_name}")
        
        # This line is parsed by app.py to track overall file progress
        print(f"Processing File {i + 1}/{total_videos}: {base_name}", flush=True)

        inpainted_reader, splatted_reader, original_reader = None, None, None
        original_video_path_to_move = None
        try:
            # Use active_settings for per-video overrideable params
            s = active_settings
            inpaint_suffix_reg = r"_inpainted_right_eyeF?\.mp4$"
            sbs_suffix_reg = r"_inpainted_sbsF?\.mp4$"

            is_sbs_input = bool(re.search(sbs_suffix_reg, base_name))
            match = re.search(sbs_suffix_reg if is_sbs_input else inpaint_suffix_reg, base_name)
            if not match:
                logger.error(f"Could not identify suffix for '{base_name}'. Skipping.")
                continue
            suffix_to_remove = match.group(0)
            core_name_with_width = base_name[: -len(suffix_to_remove)]

            last_underscore_idx = core_name_with_width.rfind("_")
            if last_underscore_idx == -1:
                logger.error(f"Could not parse core name from '{core_name_with_width}'.")
                continue
            core_name = core_name_with_width[:last_underscore_idx]

            search_folders = []
            if settings.get("inpainted_folder"):
                search_folders.append(settings.get("inpainted_folder"))
            if settings.get("original_folder"):
                search_folders.append(settings.get("original_folder"))

            clip_sidecar_data = read_clip_sidecar(sidecar_manager, inpainted_video_path, core_name, search_folders)
            flip_horizontal = clip_sidecar_data.get("flip_horizontal", False)
            if not flip_horizontal and os.path.splitext(base_name)[0].endswith("F"):
                flip_horizontal = True

            left_border = clip_sidecar_data.get("left_border", 0.0)
            right_border = clip_sidecar_data.get("right_border", 0.0)

            mask_folder = settings["mask_folder"]
            splatted4_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted4*.mp4")
            splatted2_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted2*.mp4")
            splatted4_matches = glob.glob(splatted4_pattern)
            splatted2_matches = glob.glob(splatted2_pattern)

            splatted_file_path = None
            is_dual_input = True
            if splatted4_matches:
                splatted_file_path = splatted4_matches[0]
                is_dual_input = False
            elif splatted2_matches:
                splatted_file_path = splatted2_matches[0]
                is_dual_input = True

            if not splatted_file_path or not os.path.exists(splatted_file_path):
                logger.error(f"Missing required splatted file for '{core_name}'. Skipping.")
                continue

            inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
            splatted_reader = VideoReader(splatted_file_path, ctx=cpu(0))

            original_reader = None
            if is_dual_input:
                original_video_path = find_video_by_core_name(settings["original_folder"], core_name)
                original_video_path_to_move = original_video_path

                if original_video_path and os.path.exists(original_video_path):
                    original_reader = VideoReader(original_video_path, ctx=cpu(0))
            else:
                original_reader = splatted_reader

            num_frames = len(inpainted_reader)
            fps = inpainted_reader.get_avg_fps()
            video_stream_info = get_video_stream_info(inpainted_video_path)

            sample_splatted_np = splatted_reader.get_batch([0]).asnumpy()
            _, H_splat, W_splat, _ = sample_splatted_np.shape
            if is_dual_input:
                hires_H, hires_W = H_splat, W_splat // 2
            else:
                hires_H, hires_W = H_splat // 2, W_splat // 2

            output_format = s["output_format"]
            if original_reader is None and output_format != "Right-Eye Only":
                output_format = "Right-Eye Only"

            perceived_width_for_filename = hires_W

            output_height = hires_H
            if output_format == "Full SBS Cross-eye (Right-Left)":
                output_width = hires_W * 2
                output_suffix = "_merged_full_sbsx.mp4"
            elif output_format == "Full SBS (Left-Right)":
                output_width = hires_W * 2
                output_suffix = "_merged_full_sbs.mp4"
            elif output_format == "Double SBS":
                output_width = hires_W * 2
                output_height = hires_H * 2
                output_suffix = "_merged_half_sbs.mp4"
                perceived_width_for_filename = hires_W * 2
            elif output_format == "Half SBS (Left-Right)":
                output_width = hires_W
                output_suffix = "_merged_half_sbs.mp4"
            elif output_format in ["Anaglyph (Red/Cyan)", "Anaglyph Half-Color"]:
                output_width = hires_W
                output_suffix = "_merged_anaglyph.mp4"
            else:
                output_width = hires_W
                output_suffix = "_merged_right_eye.mp4"

            output_filename = f"{core_name}_{perceived_width_for_filename}{output_suffix}"
            output_path = os.path.join(settings["output_folder"], output_filename)

            ffmpeg_process = start_ffmpeg_pipe_process(
                content_width=output_width,
                content_height=output_height,
                final_output_mp4_path=output_path,
                fps=fps,
                video_stream_info=video_stream_info,
                pad_to_16_9=s.get("pad_to_16_9", False),
                output_format_str=output_format,
                encoding_options={
                    "codec": s.get("codec", "Auto"),
                    "encoding_quality": s.get("encoding_quality", "Medium"),
                    "encoding_tune": s.get("encoding_tune", "None"),
                    "output_crf": s.get("output_crf", 23),
                    "color_tags": s.get("color_tags", "Auto"),
                    "nvenc_lookahead_enabled": s.get("nvenc_lookahead_enabled", False),
                    "nvenc_lookahead": s.get("nvenc_lookahead", 16),
                    "nvenc_spatial_aq": s.get("nvenc_spatial_aq", False),
                    "nvenc_temporal_aq": s.get("nvenc_temporal_aq", False),
                    "nvenc_aq_strength": s.get("nvenc_aq_strength", 8),
                },
            )

            if ffmpeg_process is None:
                raise RuntimeError("Failed to start FFmpeg pipe process.")

            stdout_thread = threading.Thread(
                target=_read_ffmpeg_output, args=(ffmpeg_process.stdout, logging.DEBUG), daemon=True
            )
            stderr_thread = threading.Thread(
                target=_read_ffmpeg_output, args=(ffmpeg_process.stderr, logging.DEBUG), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()

            chunk_size = s.get("batch_chunk_size", 32)

            # tqdm progress bar - parsed by app.py for real-time UI updates
            with tqdm(total=num_frames, desc=f"Merging {base_name}", file=sys.stderr) as pbar:
                for frame_start in range(0, num_frames, chunk_size):
                    frame_end = min(frame_start + chunk_size, num_frames)
                    frame_indices = list(range(frame_start, frame_end))
                    if not frame_indices:
                        break

                    inpainted_np = inpainted_reader.get_batch(frame_indices).asnumpy()
                    splatted_np = splatted_reader.get_batch(frame_indices).asnumpy()

                    inpainted_tensor_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
                    splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0
                    _, _, H, W = splatted_tensor.shape

                    if is_dual_input:
                        half_W = W // 2
                        if original_reader is None:
                            original_left = torch.zeros(inpainted_tensor_full.shape[0], 3, H, half_W)
                        else:
                            original_np = original_reader.get_batch(frame_indices).asnumpy()
                            original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0
                        mask_raw = splatted_tensor[:, :, :, :half_W]
                        warped_original = splatted_tensor[:, :, :, half_W : half_W * 2]
                    else:
                        half_H = H // 2
                        half_W = W // 2
                        original_left = splatted_tensor[:, :, :half_H, :half_W]
                        mask_raw = splatted_tensor[:, :, half_H : half_H * 2, :half_W]
                        warped_original = splatted_tensor[:, :, half_H : half_H * 2, half_W : half_W * 2]

                    # Extract inpainted right eye, using hires_W to avoid off-by-one with odd widths
                    if is_sbs_input:
                        inp_half = inpainted_tensor_full.shape[3] // 2
                        inpainted = inpainted_tensor_full[:, :, :, inp_half : inp_half + hires_W]
                    else:
                        inpainted = inpainted_tensor_full

                    if flip_horizontal and is_dual_input:
                        original_left = torch.flip(original_left, dims=[3])

                    # Resize original_left to match warped dimensions if needed
                    if original_left.shape[2] != hires_H or original_left.shape[3] != hires_W:
                        original_left = F.interpolate(
                            original_left, size=(hires_H, hires_W), mode="bicubic", align_corners=False
                        )

                    mask_np = mask_raw.permute(0, 2, 3, 1).cpu().numpy()
                    mask_gray_np = np.mean(mask_np, axis=3)
                    mask = torch.from_numpy(mask_gray_np).float().unsqueeze(1)

                    use_gpu = s.get("use_gpu", False) and torch.cuda.is_available()
                    device = "cuda" if use_gpu else "cpu"
                    
                    with torch.autocast("cuda", enabled=use_gpu, dtype=torch.float16):
                        mask = mask.to(device)
                        inpainted = inpainted.to(device)
                        original_left = original_left.to(device)
                        warped_original = warped_original.to(device)
    
                        if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                            target_aspect = hires_W / hires_H
                            inpaint_aspect = inpainted.shape[3] / inpainted.shape[2]
    
                            if abs(inpaint_aspect - target_aspect) > 0.01:
                                if inpaint_aspect > target_aspect:
                                    new_w = hires_W
                                    new_h = int(round(hires_W / inpaint_aspect))
                                else:
                                    new_h = hires_H
                                    new_w = int(round(hires_H * inpaint_aspect))
                                inpainted = F.interpolate(
                                    inpainted, size=(new_h, new_w), mode="bicubic", align_corners=False
                                )
                                mask = F.interpolate(mask, size=(new_h, new_w), mode="bilinear", align_corners=False)
                                inpainted = F.interpolate(
                                    inpainted, size=(hires_H, hires_W), mode="bicubic", align_corners=False
                                )
                                mask = F.interpolate(mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False)
                            else:
                                inpainted = F.interpolate(
                                    inpainted, size=(hires_H, hires_W), mode="bicubic", align_corners=False
                                )
                                mask = F.interpolate(mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False)
    
    
    
                        if s.get("enable_color_transfer", False):
                            adjusted_frames = []
                            for frame_idx in range(inpainted.shape[0]):
                                adjusted_frame = apply_color_transfer(
                                    original_left[frame_idx].cpu(), inpainted[frame_idx].cpu()
                                )
                                adjusted_frames.append(adjusted_frame.to(device))
                            inpainted = torch.stack(adjusted_frames)

                    processed_mask = mask.clone()
                    if s["mask_binarize_threshold"] >= 0.0:
                        processed_mask = (mask > s["mask_binarize_threshold"]).float()

                    if s["mask_dilate_kernel_size"] > 0:
                        processed_mask = apply_mask_dilation(
                            processed_mask, s["mask_dilate_kernel_size"], use_gpu
                        )
                    if s["mask_blur_kernel_size"] > 0:
                        processed_mask = apply_gaussian_blur(processed_mask, s["mask_blur_kernel_size"], use_gpu)

                    if s["shadow_shift"] > 0:
                        processed_mask = apply_shadow_blur(
                            processed_mask,
                            s["shadow_shift"],
                            float(s.get("shadow_start_opacity") or 0.7),
                            float(s.get("shadow_opacity_decay") or 0.1),
                            float(s.get("shadow_min_opacity") or 0.0),
                            float(s.get("shadow_decay_gamma") or 1.0),
                            use_gpu,
                        )

                    # Ensure inpainted and mask exactly match warped_original dims (resize, never crop)
                    t_H, t_W = warped_original.shape[2], warped_original.shape[3]
                    if inpainted.shape[2] != t_H or inpainted.shape[3] != t_W:
                        inpainted = F.interpolate(inpainted, size=(t_H, t_W), mode="bicubic", align_corners=False)
                    if processed_mask.shape[2] != t_H or processed_mask.shape[3] != t_W:
                        processed_mask = F.interpolate(processed_mask, size=(t_H, t_W), mode="bilinear", align_corners=False)

                    blended_right_eye = warped_original * (1 - processed_mask) + inpainted * processed_mask

                    # Convergence adjustment (horizontal shift of right eye)
                    convergence = int(s.get("convergence", 0))
                    if convergence != 0:
                        shifted = torch.zeros_like(blended_right_eye)
                        if convergence > 0:
                            shifted[:, :, :, convergence:] = blended_right_eye[:, :, :, :-convergence]
                        else:
                            c = -convergence
                            shifted[:, :, :, :-c] = blended_right_eye[:, :, :, c:]
                        blended_right_eye = shifted

                    if s.get("add_borders", True) and (left_border > 0 or right_border > 0):
                        original_left, blended_right_eye = apply_borders_to_frames(
                            left_border, right_border, original_left, blended_right_eye
                        )

                    if output_format == "Full SBS (Left-Right)":
                        final_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                    elif output_format == "Full SBS Cross-eye (Right-Left)":
                        final_chunk = torch.cat([blended_right_eye, original_left], dim=3)
                    elif output_format == "Half SBS (Left-Right)":
                        resized_left = F.interpolate(
                            original_left, size=(hires_H, hires_W // 2), mode="bilinear", align_corners=False
                        )
                        resized_right = F.interpolate(
                            blended_right_eye, size=(hires_H, hires_W // 2), mode="bilinear", align_corners=False
                        )
                        final_chunk = torch.cat([resized_left, resized_right], dim=3)
                    elif output_format == "Double SBS":
                        sbs_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                        final_chunk = F.interpolate(
                            sbs_chunk, size=(hires_H * 2, hires_W * 2), mode="bilinear", align_corners=False
                        )
                    elif output_format == "Anaglyph (Red/Cyan)":
                        final_chunk = torch.cat(
                            [
                                original_left[:, 0:1, :, :],
                                blended_right_eye[:, 1:3, :, :],
                            ],
                            dim=1,
                        )
                    elif output_format == "Anaglyph Half-Color":
                        left_gray = (
                            original_left[:, 0, :, :] * 0.299
                            + original_left[:, 1, :, :] * 0.587
                            + original_left[:, 2, :, :] * 0.114
                        )
                        left_gray = left_gray.unsqueeze(1)
                        final_chunk = torch.cat(
                            [
                                left_gray,
                                blended_right_eye[:, 1:3, :, :],
                            ],
                            dim=1,
                        )
                    else:
                        final_chunk = blended_right_eye

                    if flip_horizontal:
                        final_chunk = torch.flip(final_chunk, dims=[3])

                    cpu_chunk = final_chunk.float().cpu()
    
                    for frame_tensor in cpu_chunk:
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()
                        frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
                        frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)
                        ffmpeg_process.stdin.write(frame_bgr.tobytes())
    
                    pbar.update(len(frame_indices))
                    
                    # Explicit GPU Memory Cleanup
                    del inpainted_tensor_full, splatted_tensor, mask_raw
                    del mask, inpainted, original_left, warped_original, processed_mask, blended_right_eye
                    del final_chunk, cpu_chunk
                    if 'shifted' in locals():
                        del shifted
                    if 'sbs_chunk' in locals():
                        del sbs_chunk
                    if 'resized_left' in locals():
                        del resized_left
                    if 'resized_right' in locals():
                        del resized_right
                    if 'left_gray' in locals():
                        del left_gray
                    
                    if use_gpu:
                        torch.cuda.empty_cache()
                    gc.collect()

            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()

            ffmpeg_process.wait(timeout=120)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            if ffmpeg_process.returncode != 0:
                logger.error(f"FFmpeg encoding failed for {base_name}. Check console for details.")
            else:
                logger.info(f"Successfully encoded video to {output_path}")

                del ffmpeg_process
                if inpainted_reader:
                    del inpainted_reader
                if splatted_reader:
                    del splatted_reader
                if original_reader:
                    del original_reader
                inpainted_reader, splatted_reader, original_reader = (None, None, None)
                time.sleep(0.1)

                if settings.get("resume", False):
                    cleanup_files = [
                        (inpainted_video_path, settings["inpainted_folder"]),
                        (splatted_file_path, settings["mask_folder"]),
                    ]
                    if original_video_path_to_move:
                        cleanup_files.append((original_video_path_to_move, settings["original_folder"]))
                        original_base = os.path.splitext(original_video_path_to_move)[0]
                        for ext in [".fssidecar", ".json"]:
                            sidecar_path = original_base + ext
                            if os.path.exists(sidecar_path):
                                cleanup_files.append((sidecar_path, settings["original_folder"]))

                    inpainted_base = os.path.splitext(inpainted_video_path)[0]
                    for ext in [".fssidecar", ".json"]:
                        sidecar_path = inpainted_base + ext
                        if os.path.exists(sidecar_path):
                            cleanup_files.append((sidecar_path, settings["inpainted_folder"]))

                    for src_path, dest_folder in cleanup_files:
                        try:
                            finished_dir = os.path.join(dest_folder, "finished")
                            os.makedirs(finished_dir, exist_ok=True)
                            shutil.move(src_path, os.path.join(finished_dir, os.path.basename(src_path)))
                        except Exception as move_err:
                            logger.warning(f"Could not move {os.path.basename(src_path)}: {move_err}")

        except Exception as e:
            if splatted_reader:
                del splatted_reader
            if original_reader:
                del original_reader
            inpainted_reader, splatted_reader, original_reader = None, None, None
            logger.error(f"Failed to process {base_name}: {e}", exc_info=True)
            raise e
        finally:
            if inpainted_reader:
                del inpainted_reader
            if splatted_reader:
                del splatted_reader
            if original_reader:
                del original_reader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Merging Pipeline for M2SVID")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        settings = json.load(f)

    run_batch_process(settings)
