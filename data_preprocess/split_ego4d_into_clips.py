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

import os
import numpy as np
import ffmpeg
import json
import argparse
import tqdm

from m2svid.utils.video_utils import open_ffmpeg_process, read_frames_in_batches_ffmpeg, get_video_fps, get_total_frames


def split_video_into_clips(input_root, video_list, output_root, frames_per_clip=150):
    os.makedirs(output_root, exist_ok=True)

    for video_name in video_list:
        try:
            video_path = os.path.join(input_root, video_name)
            if not os.path.exists(video_path):
                print(f"Skipping {video_name}: File {video_path} does not exist.")
                continue
            else:
                print(f"Processing {video_name}")


            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            fps = get_video_fps(video_path, probe)
            total_frames = get_total_frames(video_path, probe)

            clip_index = 0
            frame_batches = read_frames_in_batches_ffmpeg(video_path, batch_size=frames_per_clip, width=width, height=height)

            for batch in tqdm.tqdm(frame_batches, total=int(total_frames // frames_per_clip)):
                clip_name = f'{video_name[:-4]}_clip_{clip_index:04d}.mp4'
                clip_file = os.path.join(output_root, clip_name)

                ffmpeg_process = open_ffmpeg_process(
                    output_path=str(clip_file),
                    width=width,
                    height=height,
                    fps=fps
                )

                try:
                    for frame in batch:
                        ffmpeg_process.stdin.write(np.ascontiguousarray(frame.astype(np.uint8)).tobytes())

                finally:
                    ffmpeg_process.stdin.close()
                    ffmpeg_process.wait()

                clip_index += 1

            print(f"Processed {video_name} into {clip_index} clips.")
        except:
            print(f"Error in {video_name}!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Divide data into n_parts and select a specific part.")
    parser.add_argument("--part", type=int, default=0, help="The part index to select (0-based).")
    parser.add_argument("--n_parts", type=int, default=1, help="The number of parts to divide the data into.")
    args = parser.parse_args()

    with open('datasets/ego4d/subsets/all_full_videos.json') as fin:
        data = json.load(fin)
    dataset_root = 'datasets/ego4d'
    video_root = f'{dataset_root}/rectified_videos_loftr_ransac'
    output_root = f'{dataset_root}/clips/rectified_videos'

    split_data = np.array_split(data, args.n_parts)
    data = split_data[args.part].tolist()

    split_video_into_clips(video_root, data, output_root, frames_per_clip=150)
