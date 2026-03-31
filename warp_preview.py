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
preprocess_depth_frame = None


def _ensure_imports():
    global _decord_loaded, VideoReader, cpu_ctx, scatter_image, preprocess_depth_frame

    if not _decord_loaded:
        from decord import VideoReader as VR, cpu
        VideoReader = VR
        cpu_ctx = cpu
        from m2svid.warping.warping import scatter_image as _si
        scatter_image = _si
        from depth_preprocess import preprocess_depth_frame as _pdf
        preprocess_depth_frame = _pdf
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
    
    Settings dict can include:
        disparity_perc: float
        preview_source: str
        dilate_x, dilate_y: float (depth dilation)
        blur_x, blur_y: int (depth blur)
        dilate_left: float (directional dilation)
        blur_left: int (edge-aware blur)
        blur_left_mix: float (H/V balance for blur_left)
        use_cuda: bool (GPU-accelerated warping)
    """
    _ensure_imports()

    video_path = video_info["video"]
    depth_path = video_info["depth"]
    disparity_perc = float(settings.get("disparity_perc", 0.035))
    preview_source = settings.get("preview_source", "Reprojected Right")

    # Depth preprocessing settings
    dilate_x = float(settings.get("dilate_x", 0.0))
    dilate_y = float(settings.get("dilate_y", 0.0))
    blur_x = int(settings.get("blur_x", 0))
    blur_y = int(settings.get("blur_y", 0))
    dilate_left = float(settings.get("dilate_left", 0.0))
    blur_left = int(settings.get("blur_left", 0))
    blur_left_mix = float(settings.get("blur_left_mix", 0.5))
    use_cuda = bool(settings.get("use_cuda", False))

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

    # Store raw depth for "Depth Map" preview
    raw_depth_gray = depth_gray.copy()

    # Apply depth preprocessing
    depth_gray = preprocess_depth_frame(
        depth_gray,
        dilate_x=dilate_x,
        dilate_y=dilate_y,
        blur_x=blur_x,
        blur_y=blur_y,
        dilate_left=dilate_left,
        blur_left=blur_left,
        blur_left_mix=blur_left_mix,
        use_cuda=use_cuda,
    )

    # Calculate disparity scale
    disparity_scale = w * disparity_perc

    # Return early for depth-only previews (no warping needed)
    if preview_source == "Depth Map (Raw)":
        depth_vis = (np.clip(raw_depth_gray, 0, 1) * 255).astype(np.uint8)
        final_frame = np.stack([depth_vis] * 3, axis=-1)
        pil_img = Image.fromarray(final_frame)
        del video_reader, depth_reader
        gc.collect()
        return pil_img, num_frames

    if preview_source == "Depth Map (Processed)":
        depth_vis = (np.clip(depth_gray, 0, 1) * 255).astype(np.uint8)
        final_frame = np.stack([depth_vis] * 3, axis=-1)
        pil_img = Image.fromarray(final_frame)
        del video_reader, depth_reader
        gc.collect()
        return pil_img, num_frames

    # Warping (using preprocessed depth)
    reproj_right, mask, _ = scatter_image(
        video_np, depth_gray, direction=-1, scale_factor=disparity_scale, reproject_depth=False, use_cuda=use_cuda
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
