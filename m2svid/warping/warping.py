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

  reproj_img[valid_y, reproj_valid_x_coords] += (
      input_frame[valid_y, valid_x]
      * (1.0 - weight_for_plus1[valid_y, valid_x])[:, None]
  )
  reproj_img_weight[valid_y, reproj_valid_x_coords] += (
      1.0 - weight_for_plus1[valid_y, valid_x]
  )[:, None]
  reproj_img[valid_y1, reproj_valid_x_coords_plus1] += (
      input_frame[valid_y1, valid_x1]
      * weight_for_plus1[valid_y1, valid_x1][:, None]
  )
  reproj_img_weight[valid_y1, reproj_valid_x_coords_plus1] += weight_for_plus1[
      valid_y1, valid_x1
  ][:, None]

  filled_pixel_mask[(reproj_img_weight != 0)[:, :, 0]] = 1
  reproj_img[reproj_img_weight != 0] /= reproj_img_weight[
      reproj_img_weight != 0
  ]

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

  t_total = (time.perf_counter() - t0) * 1000
  _WARP_FRAME_COUNT_CPU += 1
  if _WARP_FRAME_COUNT_CPU <= 20 or _WARP_FRAME_COUNT_CPU % 10 == 0:
      print(f" [Warp] CPU Frame {_WARP_FRAME_COUNT_CPU}: {t_total:.2f}ms (Warping only)")

  return reproj_img.astype(input_frame.dtype), mask, reprojected_depth


def _scatter_cupy(
    input_frame: np.ndarray,
    inverse_depth: np.ndarray,
    direction: int,
    scale_factor: float,
    inverse_ordering: bool = False,
    reproject_depth: bool = False,
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

  # Utilize += addition sequentially to ensure duplicate coordinate collision accumulation
  # identically matching original logic output.
  reproj_img[valid_y, reproj_valid_x_coords] += (
      input_frame_cp[valid_y, valid_x]
      * (1.0 - weight_for_plus1[valid_y, valid_x])[:, None]
  )
  reproj_img_weight[valid_y, reproj_valid_x_coords] += (
      1.0 - weight_for_plus1[valid_y, valid_x]
  )[:, None]
  reproj_img[valid_y1, reproj_valid_x_coords_plus1] += (
      input_frame_cp[valid_y1, valid_x1]
      * weight_for_plus1[valid_y1, valid_x1][:, None]
  )
  reproj_img_weight[valid_y1, reproj_valid_x_coords_plus1] += weight_for_plus1[
      valid_y1, valid_x1
  ][:, None]

  filled_pixel_mask = cp.zeros((h, w)).astype(bool)
  filled_pixel_mask[(reproj_img_weight != 0)[:, :, 0]] = 1

  mask_weight_nz = reproj_img_weight != 0
  reproj_img[mask_weight_nz] /= reproj_img_weight[mask_weight_nz]

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

  # Synchronize for accurate timing
  torch.cuda.synchronize()
  t_total = (time.perf_counter() - t0) * 1000
  
  _WARP_FRAME_COUNT_GPU += 1
  if _WARP_FRAME_COUNT_GPU <= 20 or _WARP_FRAME_COUNT_GPU % 10 == 0:
      # Debug logging: Check if output is zero/black
      img_mean = np.mean(res_img)
      print(f" [Warp] GPU Frame {_WARP_FRAME_COUNT_GPU}: {t_total:.2f}ms (Warp Mean: {img_mean:.4f})")

  return res_img, res_mask, res_depth


def scatter_image(
    input_frame: np.ndarray,
    inverse_depth: np.ndarray,
    direction: int,
    scale_factor: float,
    inverse_ordering: bool = False,
    reproject_depth: bool = False,
    use_cuda: bool = False,
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
      )
  else:
    return _scatter_numpy(
        input_frame,
        inverse_depth,
        direction,
        scale_factor,
        inverse_ordering,
        reproject_depth,
    )
