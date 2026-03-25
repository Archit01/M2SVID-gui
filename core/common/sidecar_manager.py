"""
Sidecar configuration manager for video processing.
Handles reading .fssidecar and .json files that store per-clip metadata
(flip_horizontal, left_border, right_border, etc.).
"""
import os
import glob
import json
import re


class SidecarConfigManager:
    """Manages sidecar configuration files for video clips."""

    def find_sidecar_file(self, video_path, core_name, search_folders):
        """
        Searches for a sidecar file (.fssidecar or .json) near the video or in search folders.
        """
        base_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        dir_name = os.path.dirname(video_path)

        # Check adjacent to video file
        for ext in [".fssidecar", ".json"]:
            p = os.path.join(dir_name, base_no_ext + ext)
            if os.path.exists(p):
                return p
            p = os.path.join(dir_name, core_name + ext)
            if os.path.exists(p):
                return p

        # Check in search folders
        for folder in search_folders:
            if not folder or not os.path.isdir(folder):
                continue
            for ext in [".fssidecar", ".json"]:
                p = os.path.join(folder, core_name + ext)
                if os.path.exists(p):
                    return p
                p = os.path.join(folder, base_no_ext + ext)
                if os.path.exists(p):
                    return p

        return None


def find_video_by_core_name(folder, core_name):
    """
    Finds a video file in the given folder whose name starts with the core_name.
    Returns the path or None.
    """
    if not folder or not os.path.isdir(folder):
        return None

    for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.wmv"]:
        # Try exact match first
        matches = glob.glob(os.path.join(folder, core_name + ext[1:]))
        if matches:
            return matches[0]

    # Try prefix match (core_name might be a prefix of the actual filename)
    for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.wmv"]:
        matches = glob.glob(os.path.join(folder, core_name + "_*" + ext[1:]))
        if matches:
            return matches[0]
        matches = glob.glob(os.path.join(folder, core_name + ".*"))
        if matches:
            return matches[0]

    return None


def find_sidecar_file(video_path, core_name, search_folders):
    """
    Convenience function wrapper around SidecarConfigManager.find_sidecar_file.
    """
    manager = SidecarConfigManager()
    return manager.find_sidecar_file(video_path, core_name, search_folders)


def read_clip_sidecar(manager, video_path, core_name, search_folders):
    """
    Reads and returns sidecar metadata for a given video clip.
    Returns a dict with keys: left_border, right_border, flip_horizontal.
    """
    sidecar_path = manager.find_sidecar_file(video_path, core_name, search_folders)
    if not sidecar_path:
        return {}

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "left_border": data.get("left_border", 0.0),
            "right_border": data.get("right_border", 0.0),
            "flip_horizontal": data.get("flip_horizontal", False),
        }
    except Exception:
        return {}
