"""
Video I/O utilities for the merging pipeline.
Handles FFmpeg pipe process creation and video stream info extraction.
"""
import subprocess
import os
import json
import logging

logger = logging.getLogger(__name__)


def get_video_stream_info(video_path):
    """
    Probes a video file using ffprobe and returns a dict of stream metadata.
    Returns dict with keys: width, height, fps, pix_fmt, color_space, color_transfer,
    color_primaries, bit_depth, codec_name.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        probe = json.loads(result.stdout)

        video_stream = None
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video":
                video_stream = s
                break

        if not video_stream:
            return {}

        # Parse FPS
        fps = 24.0
        r_frame_rate = video_stream.get("r_frame_rate", "24/1")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            if int(den) > 0:
                fps = int(num) / int(den)

        # Parse bit depth
        bit_depth = 8
        pix_fmt = video_stream.get("pix_fmt", "yuv420p")
        if "10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt:
            bit_depth = 10
        elif "12" in pix_fmt:
            bit_depth = 12

        return {
            "width": int(video_stream.get("width", 1920)),
            "height": int(video_stream.get("height", 1080)),
            "fps": fps,
            "pix_fmt": pix_fmt,
            "color_space": video_stream.get("color_space", ""),
            "color_transfer": video_stream.get("color_transfer", ""),
            "color_primaries": video_stream.get("color_primaries", ""),
            "bit_depth": bit_depth,
            "codec_name": video_stream.get("codec_name", "h264"),
        }
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
        return {}


def start_ffmpeg_pipe_process(
    content_width, content_height, final_output_mp4_path, fps,
    video_stream_info=None, pad_to_16_9=False, output_format_str="",
    encoding_options=None
):
    """
    Starts an FFmpeg subprocess that accepts raw frames via stdin pipe.
    Returns a subprocess.Popen object.
    
    Args:
        content_width: width of the raw frames being piped in
        content_height: height of the raw frames being piped in
        final_output_mp4_path: output file path
        fps: frames per second
        video_stream_info: dict from get_video_stream_info (for color metadata)
        pad_to_16_9: whether to pad output to 16:9 aspect ratio
        output_format_str: output format string (for logging)
        encoding_options: dict with codec, crf, quality settings
    """
    if encoding_options is None:
        encoding_options = {}

    os.makedirs(os.path.dirname(final_output_mp4_path) or ".", exist_ok=True)

    # Input specification
    input_pix_fmt = "bgr48le"  # 16-bit BGR from OpenCV
    
    # Determine codec
    codec_setting = encoding_options.get("codec", "Auto")
    output_crf = encoding_options.get("output_crf", 23)

    if codec_setting == "H.265" or codec_setting == "HEVC":
        vcodec = "libx265"
        output_pix_fmt = "yuv420p10le"
    elif codec_setting == "H.264" or codec_setting == "AVC":
        vcodec = "libx264"
        output_pix_fmt = "yuv420p"
    else:
        # Auto: default to H.265 for better quality
        vcodec = "libx265"
        output_pix_fmt = "yuv420p10le"

    # Calculate padded dimensions if needed
    output_width = content_width
    output_height = content_height

    vf_filters = []
    if pad_to_16_9:
        target_aspect = 16.0 / 9.0
        current_aspect = content_width / content_height
        if abs(current_aspect - target_aspect) > 0.01:
            if current_aspect < target_aspect:
                # Need horizontal padding
                output_width = int(round(content_height * target_aspect))
                if output_width % 2 != 0:
                    output_width += 1
                pad_x = (output_width - content_width) // 2
                vf_filters.append(f"pad={output_width}:{content_height}:{pad_x}:0:black")
            else:
                # Need vertical padding
                output_height = int(round(content_width / target_aspect))
                if output_height % 2 != 0:
                    output_height += 1
                pad_y = (output_height - content_height) // 2
                vf_filters.append(f"pad={content_width}:{output_height}:0:{pad_y}:black")

    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", f"{content_width}x{content_height}",
        "-r", str(fps),
        "-i", "pipe:",
    ]

    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-c:v", vcodec,
        "-pix_fmt", output_pix_fmt,
        "-crf", str(int(output_crf)),
    ]

    # Add codec-specific options
    if vcodec == "libx265":
        cmd += ["-tag:v", "hvc1"]
    
    # Add color metadata if available
    if video_stream_info:
        color_tags = encoding_options.get("color_tags", "Auto")
        if color_tags == "Auto" and video_stream_info.get("color_primaries"):
            cs = video_stream_info.get("color_space", "")
            ct = video_stream_info.get("color_transfer", "")
            cp = video_stream_info.get("color_primaries", "")
            if cs:
                cmd += ["-colorspace", cs]
            if ct:
                cmd += ["-color_trc", ct]
            if cp:
                cmd += ["-color_primaries", cp]

    cmd += ["-loglevel", "warning", final_output_mp4_path]

    logger.info(f"FFmpeg cmd: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process
    except Exception as e:
        logger.error(f"Failed to start FFmpeg: {e}")
        return None
