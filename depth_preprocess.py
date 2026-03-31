"""
Depth Map Preprocessing for M2SVID Warping Section.

Provides functions to preprocess depth maps before warping:
- Dilation (X/Y, fractional, bidirectional)
- Dilation Left (directional, left-only)
- Gaussian Blur (X/Y)
- Blur Left (edge-aware, with H/V mix)

Ported from Splating/splatting/depth_processing.py for use in the m2svid pipeline.
Works on numpy float32 2D arrays in 0-1 range. No torch dependency.
"""

import numpy as np
import cv2
import math
import os
import sys

try:
    import torch
    import kornia
    HAS_TORCH_GPU = torch.cuda.is_available()
except Exception as e:
    print(f" [Info] Torch/Kornia GPU not available for preprocess: {e}")
    HAS_TORCH_GPU = False

# We still check for cupy for general warping compatibility
try:
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False

_LAST_PREPROCESS_MODE = None
_PREPROCESS_FRAME_COUNT_GPU = 0
_PREPROCESS_FRAME_COUNT_CPU = 0


def custom_dilate(depth_2d, kernel_size_x, kernel_size_y):
    """
    Applies fractional dilation or erosion to a 2D depth map.
    
    Supports negative values for erosion. Uses 16-bit internal precision
    with bilinear interpolation between integer kernel sizes for sub-pixel control.

    Args:
        depth_2d: numpy float32 array (H, W), values in [0, 1]
        kernel_size_x: float, X kernel size. Negative = erosion.
        kernel_size_y: float, Y kernel size. Negative = erosion.
    
    Returns:
        Processed depth map, same shape, float32 [0, 1]
    """
    kx_raw = float(kernel_size_x)
    ky_raw = float(kernel_size_y)

    if abs(kx_raw) <= 1e-5 and abs(ky_raw) <= 1e-5:
        return depth_2d

    # Handle mixed sign: process each axis separately
    if (kx_raw > 0 and ky_raw < 0) or (kx_raw < 0 and ky_raw > 0):
        depth_2d = custom_dilate(depth_2d, kx_raw, 0)
        return custom_dilate(depth_2d, 0, ky_raw)

    is_erosion = kx_raw < 0 or ky_raw < 0
    kx_abs, ky_abs = abs(kx_raw), abs(ky_raw)

    def get_dilation_params(value):
        if value <= 1e-5:
            return 1, 1, 0.0
        elif value < 3.0:
            return 1, 3, (value / 3.0)
        else:
            base = 3 + 2 * int((value - 3) // 2)
            return base, base + 2, (value - base) / 2.0

    kx_low, kx_high, tx = get_dilation_params(kx_abs)
    ky_low, ky_high, ty = get_dilation_params(ky_abs)

    # Convert to 16-bit for precision
    src_img = np.ascontiguousarray(
        np.clip(depth_2d * 65535, 0, 65535).astype(np.uint16)
    )

    def do_op(k_w, k_h, img):
        if k_w <= 1 and k_h <= 1:
            return img.astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        if is_erosion:
            return cv2.erode(img, kernel, iterations=1).astype(np.float32)
        return cv2.dilate(img, kernel, iterations=1).astype(np.float32)

    is_x_int, is_y_int = (tx <= 1e-4), (ty <= 1e-4)
    if is_x_int and is_y_int:
        final_float = do_op(kx_low, ky_low, src_img)
    elif not is_x_int and is_y_int:
        final_float = (1.0 - tx) * do_op(kx_low, ky_low, src_img) + tx * do_op(kx_high, ky_low, src_img)
    elif is_x_int and not is_y_int:
        final_float = (1.0 - ty) * do_op(kx_low, ky_low, src_img) + ty * do_op(kx_low, ky_high, src_img)
    else:
        r11 = do_op(kx_low, ky_low, src_img)
        r12 = do_op(kx_low, ky_high, src_img)
        r21 = do_op(kx_high, ky_low, src_img)
        r22 = do_op(kx_high, ky_high, src_img)
        final_float = (1.0 - tx) * ((1.0 - ty) * r11 + ty * r12) + tx * ((1.0 - ty) * r21 + ty * r22)

    return np.clip(final_float / 65535.0, 0.0, 1.0).astype(np.float32)


def custom_dilate_left(depth_2d, kernel_size):
    """
    Directional dilation to the LEFT only.
    
    Uses an asymmetric kernel anchored at (0, 0) to only dilate leftward.
    Supports fractional kernel sizes via bilinear interpolation.

    Args:
        depth_2d: numpy float32 array (H, W), values in [0, 1]
        kernel_size: float, dilation amount. Negative = erosion.
    
    Returns:
        Processed depth map, same shape, float32 [0, 1]
    """
    k_raw = float(kernel_size)
    if abs(k_raw) <= 1e-5:
        return depth_2d

    is_erosion = k_raw < 0
    k_raw = abs(k_raw)

    def get_dilation_params(value):
        if value <= 1e-5:
            return 1, 1, 0.0
        elif value < 3.0:
            return 1, 3, (value / 3.0)
        else:
            base = 3 + 2 * int((value - 3) // 2)
            return base, base + 2, (value - base) / 2.0

    k_w_low, k_w_high, t = get_dilation_params(k_raw)
    k_low = int(k_w_low // 2)
    k_high = int(k_w_high // 2)

    if k_low <= 0 and k_high <= 0:
        return depth_2d

    # Convert to 16-bit
    src_img = np.ascontiguousarray(
        np.clip(depth_2d * 65535, 0, 65535).astype(np.uint16)
    )

    def do_op(k_int, img):
        if k_int <= 0:
            return img.astype(np.float32)
        k_w = int(k_int) + 1
        kernel = np.ones((1, k_w), dtype=np.uint8)
        anchor = (0, 0)
        if is_erosion:
            return cv2.erode(img, kernel, anchor=anchor, iterations=1).astype(np.float32)
        return cv2.dilate(img, kernel, anchor=anchor, iterations=1).astype(np.float32)

    src = src_img.astype(np.float32)

    if abs(t) <= 1e-4:
        out = do_op(k_low, src)
    else:
        out_low = do_op(k_low, src)
        out_high = do_op(k_high, src)
        out = (1.0 - t) * out_low + t * out_high

    return np.clip(out / 65535.0, 0.0, 1.0).astype(np.float32)


def custom_blur(depth_2d, kernel_size_x, kernel_size_y):
    """
    Applies Gaussian blur to a 2D depth map.
    
    Uses 16-bit internal precision for accuracy. Kernel sizes are
    forced to odd numbers.

    Args:
        depth_2d: numpy float32 array (H, W), values in [0, 1]
        kernel_size_x: int, X blur kernel size
        kernel_size_y: int, Y blur kernel size
    
    Returns:
        Blurred depth map, same shape, float32 [0, 1]
    """
    k_x, k_y = int(kernel_size_x), int(kernel_size_y)
    if k_x <= 0 and k_y <= 0:
        return depth_2d

    # Force odd
    k_x = k_x if k_x % 2 == 1 else k_x + 1
    k_y = k_y if k_y % 2 == 1 else k_y + 1

    # Ensure at least 1
    k_x = max(1, k_x)
    k_y = max(1, k_y)

    # Convert to 16-bit for precision
    frame_u16 = np.ascontiguousarray(
        np.clip(depth_2d * 65535, 0, 65535).astype(np.uint16)
    )
    blurred = cv2.GaussianBlur(frame_u16, (k_x, k_y), 0)
    return np.clip(blurred.astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32)


def _blur_left_edge_aware(depth_2d, blur_left_size, blur_left_mix=0.5):
    """
    Edge-aware blur applied only near left-facing depth edges.
    
    Detects depth discontinuities (edges where depth steps up going right),
    creates a mask around those edges, and applies a weighted horizontal/vertical
    blur only within that masked region.

    Args:
        depth_2d: numpy float32 array (H, W), values in [0, 1]
        blur_left_size: int, blur kernel size
        blur_left_mix: float, H/V balance (0.0 = all horizontal, 1.0 = all vertical, 0.5 = balanced)
    
    Returns:
        Processed depth map, same shape, float32 [0, 1]
    """
    if blur_left_size <= 0:
        return depth_2d

    k_blur = int(round(blur_left_size))
    k_blur = k_blur if k_blur % 2 == 1 else k_blur + 1

    # Detect left-facing edges: pixels where depth increases going right
    EDGE_STEP = 3.0 / 255.0  # Edge threshold in normalized space
    dx = depth_2d[:, 1:] - depth_2d[:, :-1]
    edge_core = dx > EDGE_STEP

    # Create edge mask with padding
    edge_mask = np.zeros_like(depth_2d, dtype=np.float32)
    edge_mask[:, 1:] = edge_core.astype(np.float32)

    # Expand edge region using max pooling approximation (dilate)
    band_half = max(1, int(math.ceil(k_blur / 4.0)))
    dilate_kernel = np.ones((1, 2 * band_half + 1), dtype=np.uint8)
    edge_band = cv2.dilate(
        (edge_mask > 0.5).astype(np.uint8),
        dilate_kernel, iterations=1
    ).astype(np.float32)

    # Smooth the alpha mask
    alpha = cv2.GaussianBlur(edge_band, (7, 1), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # H/V mix
    mix_f = max(0.0, min(1.0, float(blur_left_mix)))
    h_weight = 1.0 - mix_f
    v_weight = mix_f

    # Apply blurs
    blurred = np.zeros_like(depth_2d)
    if h_weight > 1e-6:
        blurred_h = custom_blur(depth_2d, k_blur, 1)
        blurred = blurred + blurred_h * h_weight
    if v_weight > 1e-6:
        blurred_v = custom_blur(depth_2d, 1, k_blur)
        blurred = blurred + blurred_v * v_weight

    total_weight = h_weight + v_weight
    if total_weight > 1e-6:
        blurred = blurred / total_weight

    # Blend: only apply blur in the edge region
    result = depth_2d * (1.0 - alpha) + blurred * alpha
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# --- CuPy (GPU) Implementation ---

def _custom_dilate_torch(depth_t, kx_raw, ky_raw):
    """GPU implementation using max_pool2d (dilation) and min_pool (erosion)."""
    if abs(kx_raw) <= 1e-5 and abs(ky_raw) <= 1e-5: return depth_t
    is_erosion = kx_raw < 0 or ky_raw < 0
    kx_abs, ky_abs = abs(kx_raw), abs(ky_raw)

    def get_params(val):
        if val <= 1e-5: return 1, 1, 0.0
        elif val < 3.0: return 1, 3, (val / 3.0)
        else:
            base = 3 + 2 * int((val - 3) // 2)
            return base, base + 2, (val - base) / 2.0

    kx_low, kx_high, tx = get_params(kx_abs)
    ky_low, ky_high, ty = get_params(ky_abs)

    def do_op(kw, kh, t):
        if kw <= 1 and kh <= 1: return t
        ph, pw = kh // 2, kw // 2
        if is_erosion:
            return -torch.nn.functional.max_pool2d(-t, kernel_size=(kh, kw), stride=1, padding=(ph, pw))
        return torch.nn.functional.max_pool2d(t, kernel_size=(kh, kw), stride=1, padding=(ph, pw))

    r11 = do_op(kx_low, ky_low, depth_t)
    r12 = do_op(kx_low, ky_high, depth_t)
    r21 = do_op(kx_high, ky_low, depth_t)
    r22 = do_op(kx_high, ky_high, depth_t)
    
    # Bilinear interpolation between discrete kernel sizes
    out = (1.0 - tx) * ((1.0 - ty) * r11 + ty * r12) + tx * ((1.0 - ty) * r21 + ty * r22)
    # Correct for padding mismatch if kernels were even (max_pool might shift)
    if out.shape != depth_t.shape:
        out = torch.nn.functional.interpolate(out, size=(depth_t.shape[2], depth_t.shape[3]), mode='bilinear')
    return torch.clamp(out, 0.0, 1.0)


def _custom_dilate_left_torch(depth_t, kernel_size):
    """Directional leftward dilation using shifting and pooling."""
    k_raw = float(kernel_size)
    if abs(k_raw) <= 1e-5: return depth_t
    is_erosion = k_raw < 0
    k_raw = abs(k_raw)

    def get_params(val):
        if val <= 1e-5: return 1, 1, 0.0
        elif val < 3.0: return 1, 3, (val / 3.0)
        else:
            base = 3 + 2 * int((val - 3) // 2)
            return base, base + 2, (val - base) / 2.0

    k_w_low, k_w_high, t = get_params(k_raw)
    
    def do_op(kw, tensor):
        if kw <= 1: return tensor
        # Pad right to allow 'erosion/dilation' to pull from the right (moving features left)
        pad_size = kw - 1
        x_padded = torch.nn.functional.pad(tensor, (0, pad_size, 0, 0), mode='replicate')
        if is_erosion:
            res = -torch.nn.functional.max_pool2d(-x_padded, kernel_size=(1, kw), stride=1)
        else:
            res = torch.nn.functional.max_pool2d(x_padded, kernel_size=(1, kw), stride=1)
        return res[:, :, :tensor.shape[2], :tensor.shape[3]]

    out_low = do_op(k_w_low, depth_t)
    out_high = do_op(k_w_high, depth_t)
    out = (1.0 - t) * out_low + t * out_high
    return torch.clamp(out, 0.0, 1.0)


def _custom_blur_torch(depth_tensor, k_x, k_y):
    """Gaussian blur using Kornia."""
    k_x, k_y = int(k_x), int(k_y)
    if k_x <= 0 and k_y <= 0: return depth_tensor
    k_x = k_x if k_x % 2 == 1 else k_x + 1
    k_y = k_y if k_y % 2 == 1 else k_y + 1

    def ksize_to_sigma(k):
        if k <= 1: return 0.1 # Must be slightly above 0 to prevent division-by-zero NaN in CUDA kernels
        return 0.3 * ((k - 1) * 0.5 - 1) + 0.8

    sigma_x = ksize_to_sigma(k_x)
    sigma_y = ksize_to_sigma(k_y)
    
    # Kornia gaussian_blur2d
    return kornia.filters.gaussian_blur2d(depth_tensor, (k_y, k_x), (sigma_y, sigma_x))


def _blur_left_edge_aware_torch(depth_tensor, blur_left_size, blur_left_mix=0.5):
    """Edge-aware blur using Torch/Kornia."""
    if blur_left_size <= 0: return depth_tensor
    k_blur = int(round(blur_left_size))
    k_blur = k_blur if k_blur % 2 == 1 else k_blur + 1

    # Edge detection
    EDGE_STEP = 3.0 / 255.0
    dx = depth_tensor[:, :, :, 1:] - depth_tensor[:, :, :, :-1]
    edge_mask = torch.zeros_like(depth_tensor)
    edge_mask[:, :, :, 1:] = (dx > EDGE_STEP).to(depth_tensor.dtype)

    # Dilate mask to create band
    band_half = max(1, int(math.ceil(k_blur / 4.0)))
    k_morph_w = 2 * band_half + 1
    # Use max_pool2d instead of kornia for stability
    edge_band = torch.nn.functional.max_pool2d(edge_mask, kernel_size=(1, k_morph_w), stride=1, padding=(0, k_morph_w // 2))
    
    # Smooth alpha mask
    sigma_mask = 0.3 * ((7 - 1) * 0.5 - 1) + 0.8
    # Using Gaussian Blur as requested. 
    # sigma_x is 0.1 because kernel height is 1. This prevents the NaN/black screen issue.
    alpha = kornia.filters.gaussian_blur2d(edge_band, (1, 7), (0.1, sigma_mask))
    alpha = torch.clamp(alpha, 0.0, 1.0)

    mix_f = max(0.0, min(1.0, float(blur_left_mix)))
    h_weight = 1.0 - mix_f
    v_weight = mix_f

    blurred = torch.zeros_like(depth_tensor)
    if h_weight > 1e-6:
        blurred = blurred + _custom_blur_torch(depth_tensor, k_blur, 1) * h_weight
    if v_weight > 1e-6:
        blurred = blurred + _custom_blur_torch(depth_tensor, 1, k_blur) * v_weight
    
    # Ensure result range is valid
    result = depth_tensor * (1.0 - alpha) + blurred * alpha
    return torch.clamp(result, 0.0, 1.0)


def preprocess_depth_frame(
    depth_gray,
    dilate_x=0.0,
    dilate_y=0.0,
    blur_x=0,
    blur_y=0,
    dilate_left=0.0,
    blur_left=0,
    blur_left_mix=0.5,
    use_cuda=False,
):
    """
    Main entry point: preprocesses a single depth frame.
    
    Applies operations in the correct order matching the Splatting GUI:
    1. Dilate Left (directional)
    2. Blur Left (edge-aware with H/V mix)
    3. Dilate X/Y (bidirectional)
    4. Blur X/Y (Gaussian)

    Args:
        depth_gray: numpy float32 array (H, W), values in [0, 1]
        dilate_x: float, X dilation amount (negative = erosion)
        dilate_y: float, Y dilation amount (negative = erosion)
        blur_x: int, X blur kernel size
        blur_y: int, Y blur kernel size
        dilate_left: float, left-direction dilation amount
        blur_left: int, edge-aware blur kernel size
        blur_left_mix: float, H/V balance for blur_left (0=H, 1=V, 0.5=balanced)
        use_cuda: bool, if True use GPU acceleration
    
    Returns:
        Preprocessed depth map, same shape, float32 [0, 1]
    """
    global _LAST_PREPROCESS_MODE, _PREPROCESS_FRAME_COUNT_GPU, _PREPROCESS_FRAME_COUNT_CPU
    
    # 1. Dispatch to GPU if available and requested
    if use_cuda and HAS_TORCH_GPU:
        import time
        t0 = time.perf_counter()
        
        current_mode = "CUDA (Torch)"
        if current_mode != _LAST_PREPROCESS_MODE:
            print(f" [Preprocess] Switched to {current_mode} backend.")
            _LAST_PREPROCESS_MODE = current_mode

        try:
            device = torch.device('cuda')
            # Ensure safe conversion and clear range
            t_orig = torch.from_numpy(depth_gray.copy()).to(device).float()
            depth_t = t_orig.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
            def log_stats(label, t):
                if _PREPROCESS_FRAME_COUNT_GPU == 0: # Only log first frame of session in detail
                    print(f" [Debug] {label} - Mean: {t.mean().item():.4f}, Max: {t.max().item():.4f}")

            log_stats("Input", depth_t)
            
            # --- Applying steps on GPU ---
            if abs(float(dilate_left)) > 1e-5:
                depth_t = _custom_dilate_left_torch(depth_t, float(dilate_left))
                log_stats("Dilate Left", depth_t)
            
            if int(blur_left) > 0:
                depth_t = _blur_left_edge_aware_torch(depth_t, int(blur_left), float(blur_left_mix))
                log_stats("Blur Left", depth_t)

            if abs(float(dilate_x)) > 1e-5 or abs(float(dilate_y)) > 1e-5:
                depth_t = _custom_dilate_torch(depth_t, float(dilate_x), float(dilate_y))
                log_stats("Dilate X/Y", depth_t)

            if int(blur_x) > 0 or int(blur_y) > 0:
                depth_t = _custom_blur_torch(depth_t, int(blur_x), int(blur_y))
                log_stats("Blur X/Y", depth_t)
            
            # Back to NumPy - Explicit 2D slice is safer than squeeze()
            res = depth_t[0, 0].detach().cpu().numpy()
            log_stats("Final Output", depth_t)
            
            t_total = (time.perf_counter() - t0) * 1000
            
            _PREPROCESS_FRAME_COUNT_GPU += 1
            if _PREPROCESS_FRAME_COUNT_GPU <= 20 or _PREPROCESS_FRAME_COUNT_GPU % 10 == 0:
                # Debug logging: Check if output is zero/black
                d_mean = np.mean(res)
                print(f" [Preprocess] GPU Frame {_PREPROCESS_FRAME_COUNT_GPU}: {t_total:.2f}ms (Mean: {d_mean:.4f})")
            
            return res
        except Exception as f_err:
            print(f" [Preprocess] GPU Depth Preprocess fallback: {f_err}")
            # fallback to CPU logic below
            pass

    # 2. NumPy/OpenCV Fallback
    current_mode = "NumPy"
    if current_mode != _LAST_PREPROCESS_MODE:
        print(f" [Preprocess] Switched to {current_mode} backend.")
        _LAST_PREPROCESS_MODE = current_mode
    import time
    t0 = time.perf_counter()
    # Check if any processing is needed
    has_dilate = abs(float(dilate_x)) > 1e-5 or abs(float(dilate_y)) > 1e-5
    has_blur = int(blur_x) > 0 or int(blur_y) > 0
    has_dilate_left = abs(float(dilate_left)) > 1e-5
    has_blur_left = int(blur_left) > 0

    if not (has_dilate or has_blur or has_dilate_left or has_blur_left):
        return depth_gray

    result = depth_gray.copy()

    # 1. Dilate Left
    if has_dilate_left:
        result = custom_dilate_left(result, float(dilate_left))

    # 2. Blur Left (edge-aware)
    if has_blur_left:
        result = _blur_left_edge_aware(result, int(blur_left), float(blur_left_mix))

    # 3. Dilate X/Y
    if has_dilate:
        result = custom_dilate(result, float(dilate_x), float(dilate_y))

    # 4. Blur X/Y
    if has_blur:
        result = custom_blur(result, int(blur_x), int(blur_y))

    t_total = (time.perf_counter() - t0) * 1000
    _PREPROCESS_FRAME_COUNT_CPU += 1
    if _PREPROCESS_FRAME_COUNT_CPU <= 20 or _PREPROCESS_FRAME_COUNT_CPU % 10 == 0:
        print(f" [Preprocess] CPU Frame {_PREPROCESS_FRAME_COUNT_CPU}: {t_total:.2f}ms")

    return result
