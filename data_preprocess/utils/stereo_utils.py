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
from collections import defaultdict
import numpy as np
from m2svid.warping.warping import scatter_image, scatter_image_gpu, _TORCH_CUDA_AVAILABLE


def compute_disparity(model, left_videos, right_videos):
    batch_dict = defaultdict(list)
    batch_dict["stereo_video"] = torch.stack([left_videos / 255., right_videos / 255.], dim=1)

    with torch.no_grad():
        predictions = model(batch_dict)
        disparities = predictions['raw_disparity']
        disparities = - disparities

    disparities_left = disparities

    assert disparities_left.shape[1] == 1
    disparities_left = disparities_left[:, 0].numpy()
    return disparities_left


def compute_shift(model, left_videos, right_videos, iters=20, target_min_disparity=10):
    original_left_videos = left_videos
    orignial_right_videos = right_videos

    _, _, _, width = original_left_videos.shape
    N = width // 4
    min_disparity = 0
    errors = []
    shifts = list(range(0, N, N // iters))
    min_disparities = []

    prev_error = 100000
    prev_min_disparity = 100000

    for N in shifts:
        N = N + target_min_disparity
        left_videos = original_left_videos[:,:,:, :-int(N // 2)]
        right_videos = orignial_right_videos[:,:,:, int(N // 2):]

        disparities_left = compute_disparity(model, left_videos, right_videos)

        min_disparity = np.sort(disparities_left.flatten())[:1000].mean()

        left_videos_np = left_videos.numpy().transpose(0, 2, 3, 1)
        right_videos_np = right_videos.numpy().transpose(0, 2, 3, 1)
        difference_over_frames = []
        for i in range(len(left_videos_np)):
            _scatter_fn = scatter_image_gpu if _TORCH_CUDA_AVAILABLE else scatter_image
            reprojected_right, inpainting_mask, reprojected_depth = _scatter_fn(left_videos_np[i], disparities_left[i], direction=-1, scale_factor=1, reproject_depth=True)
            difference = (np.abs(reprojected_right - right_videos_np[0]) ** 2).sum(axis=-1)
            difference = difference.flatten()[inpainting_mask.flatten() == 0]
            difference_over_frames.append(difference)
        difference_over_frames = np.concatenate(difference_over_frames)
        error = np.mean(difference_over_frames)
        errors.append(error)
        min_disparities.append(min_disparity)
        print("shift, min_disparity, error:", N, min_disparity, error)

        if min_disparity > target_min_disparity and prev_min_disparity > target_min_disparity and error < prev_error and min_disparity > prev_min_disparity:
            print("stopping serach..")
            break

        prev_error = error
        prev_min_disparity = min_disparity

    selected_shift = shifts[np.argmin(errors)] + target_min_disparity
    print("selected_shift: ", selected_shift)
    
    return selected_shift