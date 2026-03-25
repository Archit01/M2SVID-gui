"""
Image processing utilities for the merging pipeline.
Functions for mask processing, blurring, color transfer, and border application.
"""
import torch
import torch.nn.functional as F


def apply_mask_dilation(mask, kernel_size, use_gpu=False):
    """
    Applies dilation to a binary or grayscale mask using max_pool2d.
    mask: tensor of shape (B, 1, H, W)
    """
    if kernel_size <= 0:
        return mask
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    return dilated


def apply_gaussian_blur(mask, kernel_size, use_gpu=False):
    """
    Applies Gaussian blur (separable) to a mask tensor.
    mask: tensor of shape (B, 1, H, W)
    """
    if kernel_size <= 0:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Create 1D gaussian kernel
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D convolution
    kernel_h = kernel_1d.view(1, 1, kernel_size, 1).to(mask.device)
    kernel_v = kernel_1d.view(1, 1, 1, kernel_size).to(mask.device)

    padding = kernel_size // 2
    blurred = F.conv2d(mask, kernel_h, padding=(padding, 0))
    blurred = F.conv2d(blurred, kernel_v, padding=(0, padding))
    return blurred


def apply_shadow_blur(mask, shift_pixels, start_opacity, opacity_decay, min_opacity, decay_gamma, use_gpu=False):
    """
    Creates a shadow/edge mitigation effect by shifting and blurring the mask.
    Useful for hiding hard edges where the inpainting meets the original background.
    """
    if shift_pixels <= 0:
        return mask

    device = mask.device
    B, C, H, W = mask.shape

    # Create progressive shadow by shifting mask to the right
    shadow_accumulator = torch.zeros_like(mask)

    for s in range(1, shift_pixels + 1):
        shifted = torch.zeros_like(mask)
        if s < W:
            shifted[:, :, :, s:] = mask[:, :, :, :-s]

        # Calculate opacity for this shift distance
        progress = (s - 1) / max(shift_pixels - 1, 1)
        opacity = start_opacity * ((1.0 - progress) ** decay_gamma)
        opacity = max(opacity - opacity_decay * s, min_opacity)

        # Only keep shadow area not covered by original mask
        new_shadow = (shifted - mask).clamp(0, 1) * opacity
        shadow_accumulator = torch.max(shadow_accumulator, new_shadow)

    # Blur the shadow for smooth falloff
    blur_size = max(shift_pixels // 3, 3)
    if blur_size % 2 == 0:
        blur_size += 1
    shadow_accumulator = apply_gaussian_blur(shadow_accumulator, blur_size, use_gpu)

    combined = (mask + shadow_accumulator).clamp(0, 1)
    return combined


def apply_color_transfer(source, target):
    """
    Mean/std color transfer between two image tensors (C, H, W).
    source: original left eye (style reference)
    target: inpainted right eye (to be adjusted)
    Returns adjusted tensor (C, H, W).
    """
    s_mean = source.mean(dim=(1, 2), keepdim=True)
    s_std = source.std(dim=(1, 2), keepdim=True) + 1e-5

    t_mean = target.mean(dim=(1, 2), keepdim=True)
    t_std = target.std(dim=(1, 2), keepdim=True) + 1e-5

    normalized = (target - t_mean) / t_std
    adjusted = normalized * s_std + s_mean
    return adjusted.clamp(0, 1)


def apply_borders_to_frames(left_pct, right_pct, left_tensor, right_tensor):
    """
    Applies black borders to the left and right edges of both stereo frames.
    left_pct / right_pct: border size as percentage of frame width.
    """
    _, _, H, W = left_tensor.shape
    l_px = int(W * (left_pct / 100.0))
    r_px = int(W * (right_pct / 100.0))

    if l_px > 0:
        left_tensor[:, :, :, :l_px] = 0
        right_tensor[:, :, :, :l_px] = 0
    if r_px > 0:
        left_tensor[:, :, :, W - r_px:] = 0
        right_tensor[:, :, :, W - r_px:] = 0

    return left_tensor, right_tensor


def apply_dubois_anaglyph_torch(left, right):
    """
    Dubois optimized Red/Cyan anaglyph using standard matrices.
    left, right: tensors of shape (B, 3, H, W)
    Returns (B, 3, H, W) tensor.
    """
    # Dubois Red/Cyan matrix coefficients
    r = (
        left[:, 0:1] * 0.4561 + left[:, 1:2] * 0.500484 + left[:, 2:3] * 0.176381
        - right[:, 0:1] * 0.0434706 - right[:, 1:2] * 0.0879388 - right[:, 2:3] * 0.00155529
    )
    g = (
        -left[:, 0:1] * 0.0400822 - left[:, 1:2] * 0.0378246 - left[:, 2:3] * 0.0157589
        + right[:, 0:1] * 0.378476 + right[:, 1:2] * 0.73364 + right[:, 2:3] * 0.0184503
    )
    b = (
        -left[:, 0:1] * 0.0152161 - left[:, 1:2] * 0.0205971 - left[:, 2:3] * 0.00546856
        - right[:, 0:1] * 0.0721527 - right[:, 1:2] * 0.112961 + right[:, 2:3] * 1.2264
    )
    return torch.cat([r, g, b], dim=1).clamp(0, 1)


def apply_optimized_anaglyph_torch(left, right):
    """
    Optimized Anaglyph using the 'half-color' approach.
    left, right: tensors of shape (B, 3, H, W)
    Returns (B, 3, H, W) tensor.
    """
    left_gray = (
        left[:, 0:1] * 0.299
        + left[:, 1:2] * 0.587
        + left[:, 2:3] * 0.114
    )
    return torch.cat([left_gray, right[:, 1:2], right[:, 2:3]], dim=1).clamp(0, 1)
