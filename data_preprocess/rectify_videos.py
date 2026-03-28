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

import json
import numpy as np
import torch
import ffmpeg
import os
import tqdm
import traceback
import argparse

from m2svid.utils.video_utils import open_ffmpeg_process, read_frames_in_batches_ffmpeg, get_video_fps, get_total_frames, split_left_right
from utils.rectify_utils import rectify_videos, combine_videos_with_lines, torch_to_opencv_format, opencv_to_torch_format, opencv_to_numpy_format


def process_video(name, REWRITE):
    video_path = f'{video_root}/{name}'
    params_path = f'{params_root}/{name}.json'
    output_path_video = f'{output_root_video}/{name}'
    debug_output_path = f'{debug_root}/{name}' if debug_root is not None else None

    ffmpeg_process_video = None

    try:
        if os.path.exists(output_path_video):
            print(f'Video: {name} is already processed.')
            if REWRITE:
                print(f'Rewriting video: {name} .')
                os.remove(output_path_video)
            else:
                return
        else:
            print(f'Processing {name}.')
        with open(params_path) as fin:
            params = json.load(fin)

            x_min, y_min, x_max, y_max = params['croping']
            y_max = int((y_max - y_min) // 2 * 2) + y_min
            x_max = int((x_max - x_min) // 2 * 2) + x_min
            params['croping'] = x_min, y_min, x_max, y_max

            rectification_params = np.array(params['homography_left']), np.array(params['homography_right']), params['croping']

        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = get_video_fps(video_path, probe)
        total_frames = get_total_frames(video_path, probe)

        for i, frames in enumerate(tqdm.tqdm(read_frames_in_batches_ffmpeg(video_path, batch_size, width, height), total=int(total_frames // batch_size))):
            frames = frames.transpose(0, 3, 1, 2)
            frames = torch.from_numpy(frames)

            left_videos, right_videos = split_left_right(frames, rectified=False)

            if debug_output_path is not None:
                tmp = f'{debug_output_path}/matching_examples/{name}'
                combine_videos_with_lines(left_videos, right_videos, tmp)

            left_frames = torch_to_opencv_format(left_videos)
            right_frames = torch_to_opencv_format(right_videos)

            _, left_frames, right_frames = rectify_videos(left_frames, right_frames, rectification_params=rectification_params, output_folder=debug_output_path)

            if debug_output_path is not None:
                rec_left_videos = opencv_to_torch_format(left_frames)
                rec_right_videos = opencv_to_torch_format(right_frames)

                tmp = f'{debug_output_path}/matching_examples_rec/{name}'
                combine_videos_with_lines(rec_left_videos, rec_right_videos, tmp)


            left_videos = opencv_to_numpy_format(left_frames)
            right_videos = opencv_to_numpy_format(right_frames)
            video = np.concatenate((left_videos, right_videos), axis=2)
            if ffmpeg_process_video is None:
                _, height, width, _ = left_videos.shape
                if height == 0 and width == 0:
                    print(f'Error ({name}), height: {height}, width: {width}')
                    raise ValueError
                ffmpeg_process_video = open_ffmpeg_process(output_path_video, width * 2, height, fps)

            ffmpeg_process_video.stdin.write(np.ascontiguousarray(video.astype(np.uint8)).tobytes())

        ffmpeg_process_video.stdin.close()

        ffmpeg_process_video.wait()
        print(f'Processing {name} is done.')

    except Exception:
        print(f'--------------- Error processing video: {name} --------------------')
        traceback.print_exc()
    finally:
        if ffmpeg_process_video is not None:
            ffmpeg_process_video.stdin.close()
            ffmpeg_process_video.wait()


parser = argparse.ArgumentParser(description="Divide data into n_parts and select a specific part.")
parser.add_argument("--part", type=int, default=0, help="The part index to select (0-based).")
parser.add_argument("--n_parts", type=int, default=1, help="The number of parts to divide the data into.")
parser.add_argument("--batch_size", type=int, default=10, help="The number of parts to divide the data into.")
parser.add_argument("--rewrite", default=False, action='store_true')
parser.add_argument("--debug", default=False, action='store_true')

args = parser.parse_args()
batch_size = args.batch_size
REWRITE = args.rewrite
debug = args.debug


dataset_root = 'datasets/ego4d'
with open('datasets/ego4d/subsets/all_full_videos.json') as fin:
    data = json.load(fin)

video_root = f'{dataset_root}/videos/'

params_root = f'{dataset_root}/recrification_params_loftr_ransac'
output_root_video = f'{dataset_root}/rectified_videos_loftr_ransac'
if debug:
    debug_root = f'{dataset_root}/examples/rectify_videos_loftr_ransac'
else:
    debug_root = None
os.makedirs(output_root_video, exist_ok=True)


split_data = np.array_split(data, args.n_parts)
data = split_data[args.part].tolist()
print(f"Processing part {args.part} out of {args.n_parts}: {len(data)} videos")


for name in data:
    process_video(name, REWRITE)



