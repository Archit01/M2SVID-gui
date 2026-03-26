"""
Merge Preview Module - generates a single blended preview frame
for the Gradio Merging GUI. Imported directly (no subprocess) for speed.
"""
import os
import glob
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gc

# Lazy imports for heavy modules
_decord_loaded = False
_core_loaded = False
VideoReader = None
cpu_ctx = None
apply_mask_dilation = None
apply_gaussian_blur = None
apply_shadow_blur = None
apply_color_transfer = None
apply_borders_to_frames = None
SidecarConfigManager = None
find_video_by_core_name = None
read_clip_sidecar = None


def _ensure_imports():
    global _decord_loaded, _core_loaded
    global VideoReader, cpu_ctx
    global apply_mask_dilation, apply_gaussian_blur, apply_shadow_blur
    global apply_color_transfer, apply_borders_to_frames
    global SidecarConfigManager, find_video_by_core_name, read_clip_sidecar

    if not _decord_loaded:
        from decord import VideoReader as VR, cpu
        VideoReader = VR
        cpu_ctx = cpu
        _decord_loaded = True

    if not _core_loaded:
        from core.common.image_processing import (
            apply_mask_dilation as _amd, apply_gaussian_blur as _agb,
            apply_shadow_blur as _asb, apply_color_transfer as _act,
            apply_borders_to_frames as _abtf
        )
        from core.common.sidecar_manager import (
            SidecarConfigManager as _scm,
            find_video_by_core_name as _fvbcn,
            read_clip_sidecar as _rcs
        )
        apply_mask_dilation = _amd
        apply_gaussian_blur = _agb
        apply_shadow_blur = _asb
        apply_color_transfer = _act
        apply_borders_to_frames = _abtf
        SidecarConfigManager = _scm
        find_video_by_core_name = _fvbcn
        read_clip_sidecar = _rcs
        _core_loaded = True


def scan_videos(inpainted_folder, mask_folder, original_folder):
    """
    Scans folders to find valid video sets for merging.
    Returns a list of dicts with video info.
    """
    if not inpainted_folder or not os.path.isdir(inpainted_folder):
        return []

    all_mp4s = sorted(glob.glob(os.path.join(inpainted_folder, "*.mp4")))
    inpaint_pattern = re.compile(r"_inpainted_(right_eye|sbs)F?\.mp4$")
    valid_videos = [f for f in all_mp4s if inpaint_pattern.search(f)]

    video_list = []
    for inpainted_path in valid_videos:
        base_name = os.path.basename(inpainted_path)
        inpaint_suffix_reg = r"_inpainted_right_eyeF?\.mp4$"
        sbs_suffix_reg = r"_inpainted_sbsF?\.mp4$"

        is_sbs_input = bool(re.search(sbs_suffix_reg, base_name))
        match = re.search(sbs_suffix_reg if is_sbs_input else inpaint_suffix_reg, base_name)
        if not match:
            continue
        suffix_to_remove = match.group(0)
        core_name_with_width = base_name[: -len(suffix_to_remove)]

        last_underscore_idx = core_name_with_width.rfind("_")
        if last_underscore_idx == -1:
            continue
        core_name = core_name_with_width[:last_underscore_idx]

        # Check for splatted file
        if mask_folder and os.path.isdir(mask_folder):
            splatted4_matches = glob.glob(os.path.join(mask_folder, f"{core_name}_*_splatted4*.mp4"))
            splatted2_matches = glob.glob(os.path.join(mask_folder, f"{core_name}_*_splatted2*.mp4"))
        else:
            splatted4_matches = []
            splatted2_matches = []

        splatted_path = None
        is_dual_input = True
        if splatted4_matches:
            splatted_path = splatted4_matches[0]
            is_dual_input = False
        elif splatted2_matches:
            splatted_path = splatted2_matches[0]
            is_dual_input = True

        if not splatted_path:
            continue

        # Check for original (simple glob, no heavy imports needed)
        original_path = None
        if is_dual_input and original_folder and os.path.isdir(original_folder):
            for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.wmv"]:
                matches = glob.glob(os.path.join(original_folder, f"{core_name}{ext[1:]}"))
                if matches:
                    original_path = matches[0]
                    break

        video_list.append({
            "inpainted": inpainted_path,
            "splatted": splatted_path,
            "original": original_path,
            "core_name": core_name,
            "base_name": base_name,
            "is_sbs_input": is_sbs_input,
            "is_dual_input": is_dual_input,
        })

    return video_list


def generate_preview_frame(video_info, settings, frame_index=0):
    """
    Generates a single blended preview frame from the given video info and settings.
    Returns a PIL Image.
    """
    _ensure_imports()

    inpainted_path = video_info["inpainted"]
    splatted_path = video_info["splatted"]
    original_path = video_info.get("original")
    is_sbs_input = video_info["is_sbs_input"]
    is_dual_input = video_info["is_dual_input"]
    core_name = video_info["core_name"]
    base_name = video_info["base_name"]

    # Read sidecar
    sidecar_manager = SidecarConfigManager()
    search_folders = []
    if settings.get("inpainted_folder"):
        search_folders.append(settings["inpainted_folder"])
    if settings.get("original_folder"):
        search_folders.append(settings["original_folder"])
    clip_sidecar_data = read_clip_sidecar(sidecar_manager, inpainted_path, core_name, search_folders)

    flip_horizontal = clip_sidecar_data.get("flip_horizontal", False)
    if not flip_horizontal and os.path.splitext(base_name)[0].endswith("F"):
        flip_horizontal = True
    left_border = clip_sidecar_data.get("left_border", 0.0)
    right_border = clip_sidecar_data.get("right_border", 0.0)

    # Open readers
    inpainted_reader = VideoReader(inpainted_path, ctx=cpu_ctx(0))
    splatted_reader = VideoReader(splatted_path, ctx=cpu_ctx(0))

    num_frames = len(inpainted_reader)
    frame_index = max(0, min(frame_index, num_frames - 1))

    original_reader = None
    if is_dual_input and original_path and os.path.exists(original_path):
        original_reader = VideoReader(original_path, ctx=cpu_ctx(0))
    elif not is_dual_input:
        original_reader = splatted_reader

    # Load single frame
    inpainted_idx = frame_index
    splatted_idx = frame_index
    if settings.get("undo_reverse", False):
        inpainted_idx = num_frames - 1 - frame_index
        splatted_idx = num_frames - 1 - frame_index

    inpainted_np = inpainted_reader.get_batch([inpainted_idx]).asnumpy()
    splatted_np = splatted_reader.get_batch([splatted_idx]).asnumpy()

    inpainted_tensor_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
    splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0

    _, _, H, W = splatted_tensor.shape

    if is_dual_input:
        half_W = W // 2
        if original_reader is None:
            original_left = torch.zeros(1, 3, H, half_W)
        else:
            original_np = original_reader.get_batch([frame_index]).asnumpy()
            original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0
        mask_raw = splatted_tensor[:, :, :, :half_W]
        warped_original = splatted_tensor[:, :, :, half_W : half_W * 2]
    else:
        half_H = H // 2
        half_W = W // 2
        original_left = splatted_tensor[:, :, :half_H, :half_W]
        mask_raw = splatted_tensor[:, :, half_H : half_H * 2, :half_W]
        warped_original = splatted_tensor[:, :, half_H : half_H * 2, half_W : half_W * 2]

    hires_H, hires_W = warped_original.shape[2], warped_original.shape[3]

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

    mask = torch.mean(mask_raw, dim=1, keepdim=True)

    # Move to device
    use_gpu = settings.get("use_gpu", False) and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    with torch.autocast("cuda", enabled=use_gpu, dtype=torch.float16):
        mask = mask.to(device)
        inpainted = inpainted.to(device)
        original_left = original_left.to(device)
        warped_original = warped_original.to(device)
    
        # Resize inpainted if needed
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
                inpainted = F.interpolate(inpainted, size=(new_h, new_w), mode="bicubic", align_corners=False)
                mask = F.interpolate(mask, size=(new_h, new_w), mode="bilinear", align_corners=False)
            inpainted = F.interpolate(inpainted, size=(hires_H, hires_W), mode="bicubic", align_corners=False)
            mask = F.interpolate(mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False)
    
    
    
        # Color transfer
        if settings.get("enable_color_transfer", False) and original_left is not None:
            inpainted = apply_color_transfer(original_left[0].cpu(), inpainted[0].cpu()).unsqueeze(0).to(device)

    # Mask processing
    processed_mask = mask.clone()
    thresh = settings.get("mask_binarize_threshold", -1.0)
    if thresh >= 0.0:
        processed_mask = (mask > thresh).float()

    dilate_k = int(settings.get("mask_dilate_kernel_size", 0))
    if dilate_k > 0:
        processed_mask = apply_mask_dilation(processed_mask, dilate_k, use_gpu)

    blur_k = int(settings.get("mask_blur_kernel_size", 0))
    if blur_k > 0:
        processed_mask = apply_gaussian_blur(processed_mask, blur_k, use_gpu)

    shadow_s = int(settings.get("shadow_shift", 0))
    if shadow_s > 0:
        processed_mask = apply_shadow_blur(
            processed_mask, shadow_s,
            float(settings.get("shadow_start_opacity") or 0.7),
            float(settings.get("shadow_opacity_decay") or 0.1),
            float(settings.get("shadow_min_opacity") or 0.0),
            float(settings.get("shadow_decay_gamma") or 1.0),
            use_gpu,
        )

    # Ensure inpainted and mask exactly match warped_original dimensions (resize, never crop)
    target_H, target_W = warped_original.shape[2], warped_original.shape[3]
    if inpainted.shape[2] != target_H or inpainted.shape[3] != target_W:
        inpainted = F.interpolate(inpainted, size=(target_H, target_W), mode="bicubic", align_corners=False)
    if processed_mask.shape[2] != target_H or processed_mask.shape[3] != target_W:
        processed_mask = F.interpolate(processed_mask, size=(target_H, target_W), mode="bilinear", align_corners=False)

    blended_right_eye = warped_original * (1 - processed_mask) + inpainted * processed_mask

    # Convergence adjustment (horizontal shift of right eye)
    convergence = int(settings.get("convergence", 0))
    if convergence != 0:
        shifted = torch.zeros_like(blended_right_eye)
        if convergence > 0:
            shifted[:, :, :, convergence:] = blended_right_eye[:, :, :, :-convergence]
        else:
            c = -convergence
            shifted[:, :, :, :-c] = blended_right_eye[:, :, :, c:]
        blended_right_eye = shifted

    # Apply borders
    if settings.get("add_borders", True) and (left_border > 0 or right_border > 0):
        original_left, blended_right_eye = apply_borders_to_frames(
            left_border, right_border, original_left, blended_right_eye
        )

    # Assemble based on preview source
    preview_source = settings.get("preview_source", "Blended Right Eye")

    if preview_source == "Blended Right Eye":
        final_frame = blended_right_eye
    elif preview_source == "Original Left Eye":
        final_frame = original_left if original_left is not None else torch.zeros_like(blended_right_eye)
    elif preview_source == "Warped Right BG":
        final_frame = warped_original
    elif preview_source == "Inpainted Right Eye":
        final_frame = inpainted
    elif preview_source == "Processed Mask":
        final_frame = processed_mask.repeat(1, 3, 1, 1)
    elif preview_source == "Full SBS":
        if original_left is not None:
            final_frame = torch.cat([original_left, blended_right_eye], dim=3)
        else:
            final_frame = blended_right_eye
    elif preview_source == "Anaglyph":
        if original_left is not None:
            left_gray = (
                original_left[:, 0:1, :, :] * 0.299
                + original_left[:, 1:2, :, :] * 0.587
                + original_left[:, 2:3, :, :] * 0.114
            )
            final_frame = torch.cat([left_gray, blended_right_eye[:, 1:2, :, :], blended_right_eye[:, 2:3, :, :]], dim=1)
        else:
            final_frame = blended_right_eye
    else:
        final_frame = blended_right_eye

    if flip_horizontal:
        final_frame = torch.flip(final_frame, dims=[3])

    # Convert to PIL
    final_uint8 = (final_frame[0].float().permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
    pil_img = Image.fromarray(final_uint8.cpu().numpy())

    # Release file handles from VideoReader objects (critical on Windows to prevent file locks)
    if original_reader is not None and original_reader is not splatted_reader:
        del original_reader
    del inpainted_reader, splatted_reader

    # Cleanup GPU tensors
    del mask, inpainted, original_left, warped_original, processed_mask, blended_right_eye, final_frame
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()

    return pil_img, num_frames
