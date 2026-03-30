"""
Warping Preview Module - generates a single reprojected preview frame
for the Gradio Warping GUI. Imported directly (no subprocess) for speed.
"""
import os
import glob
import numpy as np
import cv2
from PIL import Image
import gc

# Lazy imports for heavy modules
_decord_loaded = False
VideoReader = None
cpu_ctx = None
scatter_image = None
scatter_image_gpu = None
_use_gpu_scatter = False


def _ensure_imports():
    global _decord_loaded, VideoReader, cpu_ctx, scatter_image, scatter_image_gpu, _use_gpu_scatter

    if not _decord_loaded:
        from decord import VideoReader as VR, cpu
        VideoReader = VR
        cpu_ctx = cpu
        from m2svid.warping.warping import scatter_image as _si, scatter_image_gpu as _si_gpu, _TORCH_CUDA_AVAILABLE
        scatter_image = _si
        scatter_image_gpu = _si_gpu
        _use_gpu_scatter = _TORCH_CUDA_AVAILABLE
        _decord_loaded = True


def scan_videos(input_folder, depth_folder):
    """
    Scans folders to find valid video sets for warping.
    Expects input_folder to have {name}.mp4 and depth_folder to have {name}_depth.mp4
    Returns a list of dicts with video info.
    """
    if not input_folder or not os.path.isdir(input_folder):
        return []

    all_mp4s = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    video_list = []
    
    for video_path in all_mp4s:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check for depth file
        depth_path = None
        if depth_folder and os.path.isdir(depth_folder):
            # Try specific suffix first
            potential_depth = os.path.join(depth_folder, f"{base_name}_depth.mp4")
            if os.path.exists(potential_depth):
                depth_path = potential_depth
            else:
                # Try generic name if depth folder is dedicated
                potential_depth_generic = os.path.join(depth_folder, f"{base_name}.mp4")
                if os.path.exists(potential_depth_generic):
                    depth_path = potential_depth_generic

        if not depth_path:
            continue

        video_list.append({
            "video": video_path,
            "depth": depth_path,
            "base_name": base_name,
        })

    return video_list


def generate_preview_frame(video_info, settings, frame_index=0):
    """
    Generates a single warped preview frame from the given video info and settings.
    Returns a PIL Image and total_frames count.
    """
    _ensure_imports()

    video_path = video_info["video"]
    depth_path = video_info["depth"]
    disparity_perc = float(settings.get("disparity_perc", 0.035))
    preview_source = settings.get("preview_source", "Reprojected Right")

    # Open readers
    video_reader = VideoReader(video_path, ctx=cpu_ctx(0))
    depth_reader = VideoReader(depth_path, ctx=cpu_ctx(0))

    num_frames = min(len(video_reader), len(depth_reader))
    frame_index = max(0, min(frame_index, num_frames - 1))

    # Load single frame
    video_np = video_reader.get_batch([frame_index]).asnumpy()[0]
    depth_np = depth_reader.get_batch([frame_index]).asnumpy()[0]

    h, w = video_np.shape[:2]
    
    # Take red channel of depth and normalize
    depth_gray = depth_np[..., 0].astype(np.float32) / 255.0
    
    # Resize depth to match video if needed
    if depth_gray.shape[0] != h or depth_gray.shape[1] != w:
        depth_gray = cv2.resize(depth_gray, (w, h), interpolation=cv2.INTER_CUBIC)

    # Calculate disparity scale
    disparity_scale = w * disparity_perc

    # Warping
    # scatter_image(input_frame, inverse_depth, direction, scale_factor, ...)
    # direction=-1 means Right Eye reprojection (standard)
    _scatter_fn = scatter_image_gpu if _use_gpu_scatter else scatter_image
    reproj_right, mask, _ = _scatter_fn(
        video_np, depth_gray, direction=-1, scale_factor=disparity_scale, reproject_depth=False
    )

    # Assemble preview
    if preview_source == "Reprojected Right":
        final_frame = reproj_right
    elif preview_source == "Original Left":
        final_frame = video_np
    elif preview_source == "Inpainting Mask":
        # Convert 0-255 mask to RGB
        final_frame = np.stack([mask] * 3, axis=-1)
    elif preview_source == "Side-by-Side":
        final_frame = np.concatenate([video_np, reproj_right], axis=1)
    elif preview_source == "Top-Bottom (Mask/Warp)":
        mask_rgb = np.stack([mask] * 3, axis=-1)
        final_frame = np.concatenate([mask_rgb, reproj_right], axis=0)
    else:
        final_frame = reproj_right

    # Convert to PIL
    pil_img = Image.fromarray(final_frame.astype(np.uint8))

    # Release file handles
    del video_reader, depth_reader
    gc.collect()

    return pil_img, num_frames
