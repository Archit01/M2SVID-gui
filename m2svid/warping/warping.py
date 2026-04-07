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

import numpy as np
import os
import sys

# ── Bootstrap CUDA 12 DLLs from PyTorch's lib directory ──────────────────────
# On Windows, cupy-cuda12x needs cublas64_12.dll, nvrtc64_120_0.dll etc.
# If the system CUDA_PATH points to an older CUDA (e.g. 11.8), CuPy won't
# find them. PyTorch cu128 bundles these DLLs, so we add its lib dir.
if sys.platform == "win32":
    try:
        import torch
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
    except (ImportError, AttributeError, OSError):
        pass

# ── Try importing CuPy ───────────────────────────────────────────────────────
try:
    import cupy as cp
    # Quick runtime sanity check: verify GPU JIT path works
    _test = cp.array([1.0, 2.0, 3.0])
    _test_sum = float(cp.sum(_test))
    assert _test_sum == 6.0
    del _test, _test_sum
    HAS_CUPY = True
except Exception as e:
    print(f" [Info] CuPy not available for warping acceleration: {e}")
    HAS_CUPY = False

_LAST_WARP_MODE = None
_WARP_FRAME_COUNT_GPU = 0
_WARP_FRAME_COUNT_CPU = 0
_PREPROCESS_FRAME_COUNT_CPU = 0


def _scatter_numpy(
    input_frame: np.ndarray,
    inverse_depth: np.ndarray,
    direction: int,
    scale_factor: float,
    inverse_ordering: bool = False,
    reproject_depth: bool = False,
    close_micro_holes: bool = False,
    micro_hole_iters: float = 1.0,
):
  global _WARP_FRAME_COUNT_CPU
  import time
  t0 = time.perf_counter()

  h, w = input_frame.shape[:2]
  disparity_map = (inverse_depth.astype(np.float32) * scale_factor).astype(
      np.float32
  )
  disparity_map_int = disparity_map.astype(np.int32)
  weight_for_plus1 = disparity_map - disparity_map_int.astype(np.float32)
  disparity_map_int_plus1 = (disparity_map + 1.0).astype(np.int32)

  x_coords, _ = np.meshgrid(np.arange(w), np.arange(h))
  reproj_x_coords = x_coords + (disparity_map_int * direction)
  reproj_x_coords_plus1 = x_coords + (disparity_map_int_plus1 * direction)

  reproj_img = np.zeros_like(input_frame).astype(np.float32)
  reproj_img_weight = np.zeros_like(input_frame).astype(np.float32)
  filled_pixel_mask = np.zeros((h, w)).astype(bool)

  valid_mask = (reproj_x_coords >= 0) & (reproj_x_coords < w)
  valid_y, valid_x = np.where(valid_mask)
  reproj_valid_x_coords = reproj_x_coords[valid_y, valid_x]
  valid_mask1 = (reproj_x_coords_plus1 >= 0) & (reproj_x_coords_plus1 < w)
  valid_y1, valid_x1 = np.where(valid_mask1)
  reproj_valid_x_coords_plus1 = reproj_x_coords_plus1[valid_y1, valid_x1]

  if inverse_ordering:
    valid_y = valid_y[::-1]
    valid_x = valid_x[::-1]
    reproj_valid_x_coords = reproj_valid_x_coords[::-1]
    valid_y1 = valid_y1[::-1]
    valid_x1 = valid_x1[::-1]
    reproj_valid_x_coords_plus1 = reproj_valid_x_coords_plus1[::-1]

  depth_weight = np.power(1.414, disparity_map - disparity_map.min())

  w0 = (1.0 - weight_for_plus1[valid_y, valid_x]) * depth_weight[valid_y, valid_x]
  w1 = weight_for_plus1[valid_y1, valid_x1] * depth_weight[valid_y1, valid_x1]

  reproj_img[valid_y, reproj_valid_x_coords] += (
      input_frame[valid_y, valid_x] * w0[:, None]
  )
  reproj_img_weight[valid_y, reproj_valid_x_coords] += w0[:, None]

  reproj_img[valid_y1, reproj_valid_x_coords_plus1] += (
      input_frame[valid_y1, valid_x1] * w1[:, None]
  )
  reproj_img_weight[valid_y1, reproj_valid_x_coords_plus1] += w1[:, None]

  filled_pixel_mask[(reproj_img_weight != 0)[:, :, 0]] = 1
  reproj_img[reproj_img_weight != 0] /= reproj_img_weight[
      reproj_img_weight != 0
  ]

  if close_micro_holes:
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = filled_pixel_mask.astype(np.uint8)
    
    iters_floor = int(np.floor(micro_hole_iters))
    frac = float(micro_hole_iters - iters_floor)
    total_iters = iters_floor + 1 if frac > 0 else iters_floor
    
    if iters_floor > 0:
        closed_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iters_floor)
        dilated_img = cv2.dilate(reproj_img, kernel, iterations=iters_floor)
    else:
        closed_mask = mask_u8.copy()
        dilated_img = reproj_img.copy()

    if frac > 0:
        closed_mask_ceil = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=total_iters)
        dilated_img_ceil = cv2.dilate(reproj_img, kernel, iterations=total_iters)
        
        new_holes = (closed_mask_ceil == 1) & (closed_mask == 0)
        
        bayer_4x4 = np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ]) / 16.0
        y_idx, x_idx = np.indices((h, w))
        bayer_tile = bayer_4x4[y_idx % 4, x_idx % 4]
        
        frac_mask = new_holes & (bayer_tile < frac)
        
        closed_mask[frac_mask] = 1
        dilated_img[frac_mask] = dilated_img_ceil[frac_mask]

    holes_to_fill = (closed_mask == 1) & (~filled_pixel_mask)
    if np.any(holes_to_fill):
      reproj_img[holes_to_fill] = dilated_img[holes_to_fill]
    filled_pixel_mask = closed_mask.astype(bool)

  if reproject_depth:
    depth = 1 / (inverse_depth + 1e-6)
    reprojected_depth = np.zeros_like(depth, dtype=np.float32)
    reprojected_depth_weight = np.zeros_like(depth, dtype=np.float32)

    reprojected_depth[valid_y, reproj_valid_x_coords] += depth[
        valid_y, valid_x
    ] * (1.0 - weight_for_plus1[valid_y, valid_x])
    reprojected_depth_weight[valid_y, reproj_valid_x_coords] += (
        1.0 - weight_for_plus1[valid_y, valid_x]
    )
    reprojected_depth[valid_y1, reproj_valid_x_coords_plus1] += (
        depth[valid_y1, valid_x1] * weight_for_plus1[valid_y1, valid_x1]
    )
    reprojected_depth_weight[
        valid_y1, reproj_valid_x_coords_plus1
    ] += weight_for_plus1[valid_y1, valid_x1]

    reprojected_depth[
        reprojected_depth_weight != 0
    ] /= reprojected_depth_weight[reprojected_depth_weight != 0]
  else:
    reprojected_depth = None

  black_y, black_new_x = np.where(filled_pixel_mask == 0)
  black_pixel_indexes = np.ravel_multi_index(
      (black_y, black_new_x), dims=(h, w)
  )
  mask = np.zeros(input_frame.shape[:2], dtype=np.uint8)
  if black_pixel_indexes.ndim == 1:
    y_coords, x_coords = np.divmod(black_pixel_indexes, w)
  else:
    y_coords, x_coords = black_pixel_indexes[:, 0], black_pixel_indexes[:, 1]
  mask[y_coords, x_coords] = 255

  _WARP_FRAME_COUNT_CPU += 1
  # Removed per-frame timing for cleaner logs

  return reproj_img.astype(input_frame.dtype), mask, reprojected_depth


def _scatter_cupy(
    input_frame: np.ndarray,
    inverse_depth: np.ndarray,
    direction: int,
    scale_factor: float,
    inverse_ordering: bool = False,
    reproject_depth: bool = False,
    close_micro_holes: bool = False,
    micro_hole_iters: float = 1.0,
):
  h, w = input_frame.shape[:2]
  
  global _WARP_FRAME_COUNT_GPU
  
  import time
  t0 = time.perf_counter()

  input_frame_cp = cp.asarray(input_frame)
  inverse_depth_cp = cp.asarray(inverse_depth)

  disparity_map = (inverse_depth_cp.astype(cp.float32) * scale_factor).astype(cp.float32)
  disparity_map_int = disparity_map.astype(cp.int32)
  weight_for_plus1 = disparity_map - disparity_map_int.astype(cp.float32)
  disparity_map_int_plus1 = (disparity_map + 1.0).astype(cp.int32)

  x_coords, _ = cp.meshgrid(cp.arange(w), cp.arange(h))
  reproj_x_coords = x_coords + (disparity_map_int * direction)
  reproj_x_coords_plus1 = x_coords + (disparity_map_int_plus1 * direction)

  reproj_img = cp.zeros_like(input_frame_cp).astype(cp.float32)
  reproj_img_weight = cp.zeros_like(input_frame_cp).astype(cp.float32)

  valid_mask = (reproj_x_coords >= 0) & (reproj_x_coords < w)
  valid_y, valid_x = cp.where(valid_mask)
  reproj_valid_x_coords = reproj_x_coords[valid_y, valid_x]

  valid_mask1 = (reproj_x_coords_plus1 >= 0) & (reproj_x_coords_plus1 < w)
  valid_y1, valid_x1 = cp.where(valid_mask1)
  reproj_valid_x_coords_plus1 = reproj_x_coords_plus1[valid_y1, valid_x1]

  if inverse_ordering:
    valid_y = valid_y[::-1]
    valid_x = valid_x[::-1]
    reproj_valid_x_coords = reproj_valid_x_coords[::-1]
    valid_y1 = valid_y1[::-1]
    valid_x1 = valid_x1[::-1]
    reproj_valid_x_coords_plus1 = reproj_valid_x_coords_plus1[::-1]

  # Fast parallel algorithm resolving numpy overwrite mechanics
  def filter_last_occurrences(y, x):
    flat_idx = y * w + x
    _, first_occ = cp.unique(flat_idx[::-1], return_index=True)
    actual_indices = len(flat_idx) - 1 - first_occ
    return actual_indices

  keep_idx = filter_last_occurrences(valid_y, reproj_valid_x_coords)
  valid_y = valid_y[keep_idx]
  valid_x = valid_x[keep_idx]
  reproj_valid_x_coords = reproj_valid_x_coords[keep_idx]

  keep_idx1 = filter_last_occurrences(valid_y1, reproj_valid_x_coords_plus1)
  valid_y1 = valid_y1[keep_idx1]
  valid_x1 = valid_x1[keep_idx1]
  reproj_valid_x_coords_plus1 = reproj_valid_x_coords_plus1[keep_idx1]

  depth_weight = cp.power(1.414, disparity_map - disparity_map.min())

  w0 = (1.0 - weight_for_plus1[valid_y, valid_x]) * depth_weight[valid_y, valid_x]
  w1 = weight_for_plus1[valid_y1, valid_x1] * depth_weight[valid_y1, valid_x1]

  # Utilize += addition sequentially to ensure duplicate coordinate collision accumulation
  # identically matching original logic output.
  reproj_img[valid_y, reproj_valid_x_coords] += (
      input_frame_cp[valid_y, valid_x] * w0[:, None]
  )
  reproj_img_weight[valid_y, reproj_valid_x_coords] += w0[:, None]

  reproj_img[valid_y1, reproj_valid_x_coords_plus1] += (
      input_frame_cp[valid_y1, valid_x1] * w1[:, None]
  )
  reproj_img_weight[valid_y1, reproj_valid_x_coords_plus1] += w1[:, None]

  filled_pixel_mask = cp.zeros((h, w)).astype(bool)
  filled_pixel_mask[(reproj_img_weight != 0)[:, :, 0]] = 1

  mask_weight_nz = reproj_img_weight != 0
  reproj_img[mask_weight_nz] /= reproj_img_weight[mask_weight_nz]

  if close_micro_holes:
    iters_floor = int(cp.floor(micro_hole_iters).get())
    frac = float(micro_hole_iters - iters_floor)
    total_iters = iters_floor + (1 if frac > 0 else 0)
    
    mask_float = filled_pixel_mask.astype(cp.float32)
    dilated_mask = mask_float
    for _ in range(iters_floor):
      padded_mask = cp.pad(dilated_mask, pad_width=1, mode='edge')
      shifts_dil = [padded_mask[i:i+h, j:j+w] for i in range(3) for j in range(3)]
      dilated_mask = cp.max(cp.stack(shifts_dil, axis=-1), axis=-1)
    
    closed_mask = dilated_mask
    for _ in range(iters_floor):
      padded_dil = cp.pad(closed_mask, pad_width=1, mode='edge')
      shifts_ero = [padded_dil[i:i+h, j:j+w] for i in range(3) for j in range(3)]
      closed_mask = cp.min(cp.stack(shifts_ero, axis=-1), axis=-1)
    closed_mask = closed_mask > 0.5
    
    dilated_img = reproj_img
    for _ in range(iters_floor):
      padded_img = cp.pad(dilated_img, pad_width=((1, 1), (1, 1), (0, 0)), mode='edge')
      shifts_img = [padded_img[i:i+h, j:j+w, :] for i in range(3) for j in range(3)]
      dilated_img = cp.max(cp.stack(shifts_img, axis=-1), axis=-1)

    if frac > 0:
      dilated_mask_ceil = mask_float
      for _ in range(total_iters):
        padded_mask = cp.pad(dilated_mask_ceil, pad_width=1, mode='edge')
        shifts_dil = [padded_mask[i:i+h, j:j+w] for i in range(3) for j in range(3)]
        dilated_mask_ceil = cp.max(cp.stack(shifts_dil, axis=-1), axis=-1)
      
      closed_mask_ceil = dilated_mask_ceil
      for _ in range(total_iters):
        padded_dil = cp.pad(closed_mask_ceil, pad_width=1, mode='edge')
        shifts_ero = [padded_dil[i:i+h, j:j+w] for i in range(3) for j in range(3)]
        closed_mask_ceil = cp.min(cp.stack(shifts_ero, axis=-1), axis=-1)
      closed_mask_ceil = closed_mask_ceil > 0.5
      
      dilated_img_ceil = reproj_img
      for _ in range(total_iters):
        padded_img = cp.pad(dilated_img_ceil, pad_width=((1, 1), (1, 1), (0, 0)), mode='edge')
        shifts_img = [padded_img[i:i+h, j:j+w, :] for i in range(3) for j in range(3)]
        dilated_img_ceil = cp.max(cp.stack(shifts_img, axis=-1), axis=-1)

      new_holes = closed_mask_ceil & (~closed_mask)
      
      bayer_4x4_cp = cp.array([
          [ 0,  8,  2, 10],
          [12,  4, 14,  6],
          [ 3, 11,  1,  9],
          [15,  7, 13,  5]
      ]) / 16.0
      y_idx, x_idx = cp.indices((h, w))
      bayer_tile = bayer_4x4_cp[y_idx % 4, x_idx % 4]
      
      frac_mask = new_holes & (bayer_tile < frac)
      closed_mask = closed_mask | frac_mask
      
      frac_mask_3c = cp.expand_dims(frac_mask, axis=-1)
      dilated_img = cp.where(frac_mask_3c, dilated_img_ceil, dilated_img)

    holes_to_fill = closed_mask & (~filled_pixel_mask)
    filled_pixel_mask = closed_mask

    if cp.any(holes_to_fill):
      reproj_img[holes_to_fill] = dilated_img[holes_to_fill]

  if reproject_depth:
    depth = 1.0 / (inverse_depth_cp + 1e-6)
    reprojected_depth = cp.zeros_like(depth, dtype=cp.float32)
    reprojected_depth_weight = cp.zeros_like(depth, dtype=cp.float32)

    reprojected_depth[valid_y, reproj_valid_x_coords] += depth[
        valid_y, valid_x
    ] * (1.0 - weight_for_plus1[valid_y, valid_x])
    reprojected_depth_weight[valid_y, reproj_valid_x_coords] += (
        1.0 - weight_for_plus1[valid_y, valid_x]
    )
    reprojected_depth[valid_y1, reproj_valid_x_coords_plus1] += (
        depth[valid_y1, valid_x1] * weight_for_plus1[valid_y1, valid_x1]
    )
    reprojected_depth_weight[
        valid_y1, reproj_valid_x_coords_plus1
    ] += weight_for_plus1[valid_y1, valid_x1]

    mask_depth_nz = reprojected_depth_weight != 0
    reprojected_depth[mask_depth_nz] /= reprojected_depth_weight[mask_depth_nz]
  else:
    reprojected_depth = None

  black_pixel_indexes = cp.where(filled_pixel_mask == 0)
  black_y, black_new_x = black_pixel_indexes[0], black_pixel_indexes[1]

  mask = cp.zeros(input_frame_cp.shape[:2], dtype=cp.uint8)
  mask[black_y, black_new_x] = 255

  res_img = reproj_img.get()
  res_mask = mask.get()
  res_depth = reprojected_depth.get() if reprojected_depth is not None else None

  _WARP_FRAME_COUNT_GPU += 1
  # Removed per-frame timing for cleaner logs

  return res_img, res_mask, res_depth


def scatter_image(
    input_frame: np.ndarray,
    inverse_depth: np.ndarray,
    direction: int,
    scale_factor: float,
    inverse_ordering: bool = False,
    reproject_depth: bool = False,
    use_cuda: bool = False,
    close_micro_holes: bool = False,
    micro_hole_iters: float = 1.0,
):
  """Scatter-based image reprojection using depth.

  Args:
    use_cuda: If True and CuPy is available, use GPU-accelerated path.
              Falls back to NumPy if CuPy is unavailable or GPU OOMs.
  """
  global _LAST_WARP_MODE
  
  current_mode = "CUDA" if (use_cuda and HAS_CUPY) else "NumPy"
  if current_mode != _LAST_WARP_MODE:
    if current_mode == "CUDA":
        m_free, m_total = cp.cuda.Device(0).mem_info
        d_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        print(f" [Warp] Switched to CUDA acceleration on GPU: {d_name}")
        print(f" [Warp] VRAM: {m_free/1024**2:.1f}MB free / {m_total/1024**2:.1f}MB total")
    else:
        print(" [Warp] Switched to NumPy (CPU) backend.")
    _LAST_WARP_MODE = current_mode

  if use_cuda and HAS_CUPY:
    try:
      return _scatter_cupy(
          input_frame,
          inverse_depth,
          direction,
          scale_factor,
          inverse_ordering,
          reproject_depth,
          close_micro_holes,
          micro_hole_iters,
      )
    except cp.cuda.memory.OutOfMemoryError:
      print(" [Warp] !! GPU Out of Memory !! Falling back to NumPy (CPU)...")
      return _scatter_numpy(
          input_frame,
          inverse_depth,
          direction,
          scale_factor,
          inverse_ordering,
          reproject_depth,
          close_micro_holes,
          micro_hole_iters,
      )
  else:
    return _scatter_numpy(
        input_frame,
        inverse_depth,
        direction,
        scale_factor,
        inverse_ordering,
        reproject_depth,
        close_micro_holes,
        micro_hole_iters,
    )
