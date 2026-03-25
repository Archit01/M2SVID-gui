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
import numpy as np
from torchvision import transforms
from einops import rearrange
import torch
import ffmpeg
import os
import tqdm
import traceback
from multiprocessing import Pool
import gc
import cv2
import imageio
import torch as th


def read_frames_in_batches_decord(vr, batch_size, start_frame=0):
    total_frames = len(vr)
    current_frame = start_frame

    while current_frame < total_frames:
        end_frame = min(current_frame + batch_size, total_frames)
        frames = vr.get_batch(range(current_frame, end_frame)).asnumpy()
        yield frames
        current_frame = end_frame


def open_ffmpeg_process(output_path, width, height, fps, grayscale=False, no_compression=False, crf=16, bit_depth=8):
    if grayscale:
        input_pix_fmt = 'gray'
        output_pix_fmt = 'gray'
    else:
        input_pix_fmt = 'rgb24'
        output_pix_fmt = 'yuv420p10le' if bit_depth == 10 else 'yuv420p'

    if no_compression:
        ffmpeg_process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{width}x{height}', framerate=fps)
            .output(output_path, pix_fmt=output_pix_fmt, vcodec='libx264', crf=0)
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdin=True)
        )
    else:
        if crf is not None:
            ffmpeg_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{width}x{height}', framerate=fps)
                .output(output_path, pix_fmt=output_pix_fmt, vcodec='libx264', crf=crf)
                .global_args('-loglevel', 'error')
                .run_async(pipe_stdin=True)
            )
        else:
            ffmpeg_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{width}x{height}', framerate=fps)
                .output(output_path, pix_fmt=output_pix_fmt, vcodec='libx264')
                .global_args('-loglevel', 'error')
                .run_async(pipe_stdin=True)
            )
    return ffmpeg_process


def read_frames_in_batches_ffmpeg(video_path, batch_size, width, height, start_sec=0):
    process = (
        ffmpeg
        .input(video_path, ss=start_sec)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .global_args('-loglevel', 'error')
        .run_async(pipe_stdout=True)
    )

    frame_size = width * height * 3  # 3 bytes per pixel (RGB)
    buffer = bytearray()
    try:
        while True:
            chunk = process.stdout.read(frame_size * batch_size)
            if not chunk:
                break
            buffer += chunk

            while len(buffer) >= frame_size * batch_size:
                batch = np.frombuffer(buffer[:frame_size * batch_size], np.uint8)
                buffer = buffer[frame_size * batch_size:]
                batch = batch.reshape((batch_size, height, width, 3))
                yield batch

        if len(buffer) >= frame_size:
            remaining_frames = len(buffer) // frame_size
            batch = np.frombuffer(buffer[:frame_size * remaining_frames], np.uint8)
            batch = batch.reshape((remaining_frames, height, width, 3))
            yield batch
    finally:
        process.stdout.close()
        process.wait()


def get_video_fps(video_path, probe):
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream and 'r_frame_rate' in video_stream:
        # `r_frame_rate` is a fraction, e.g., '30000/1001' for ~29.97 FPS
        num, denom = map(int, video_stream['r_frame_rate'].split('/'))
        return num / denom
    else:
        raise ValueError(f"Could not determine FPS for video: {video_path}")


def get_total_frames(video_path, probe):
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = get_video_fps(video_path, probe)

    if 'nb_frames' in video_stream:
        total_frames = int(video_stream['nb_frames'])
    elif 'duration' in video_stream:
        duration = float(video_stream['duration'])
        total_frames = int(fps * duration)
    elif 'duration' in probe['format']:
        duration = float(probe['format']['duration'])
        total_frames = int(fps * duration)
    else:
        total_frames = 1
    return total_frames


def get_video_frames(video_path, fps=None, num_frames=None, width=None, height=None, duration=None,
                     start=None):
    try:
        if (width is None) or (height is None) or (duration is None):
            probe = ffmpeg.probe(video_path)

        if duration is None:
            duration = float(probe['format']['duration'])
            if duration == 0:
                duration = num_frames

        if width is None:
            width = int(probe['streams'][0]['width'])

        if height is None:
            height = int(probe['streams'][0]['height'])

    except Exception as excep:
        print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)

    try:
        if start is None:
            if num_frames is not None and fps is not None:
                num_sec = num_frames / fps
                cmd = ffmpeg.input(video_path, t=num_sec + 0.1)
            else:
                cmd = ffmpeg.input(video_path)
        else:
            assert fps is not None
            assert num_frames is not None
            num_sec = num_frames / fps
            cmd = ffmpeg.input(video_path, ss=start, t=num_sec + 0.1)

        if fps is not None:
            cmd = cmd.filter('fps', fps=fps)

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = video[:num_frames]
    except Exception as excep:
        print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)
        video = np.zeros((1, 224, 224, 3), dtype=np.uint8)

    return video


def split_left_right(frames, rectified=False, label=None):
    if not rectified:
        _, _, original_h, original_w = frames.shape
        if original_h > original_w:
            transform = transforms.Resize([original_w, original_h], antialias=True)
            frames = transform(frames)

        _, _, h, w = frames.shape
        size = int(w // 2)
        left_videos = frames[:, :, :, :size]
        right_videos = frames[:, :, :, size:]
    else:
        _, _, h, w = frames.shape
        size = int(w // 2)
        left_videos = frames[:, :, :, :size]
        right_videos = frames[:, :, :, size:]
    return left_videos, right_videos


def save_disparity_as_png(disparity, output_path):
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    disparity_16bit = np.uint16(disparity_normalized)
    cv2.imwrite(output_path, disparity_16bit)


def recover_disparity_from_png(png_path, original_min, original_max):
    disparity_16bit = imageio.imread(png_path).astype(np.float32)
    disparity_recovered = (disparity_16bit / 65535.0) * (original_max - original_min) + original_min
    return disparity_recovered



