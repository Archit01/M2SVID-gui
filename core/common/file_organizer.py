"""
File organizer utilities for the merging pipeline.
Handles moving processed files to 'finished' subdirectories.
"""
import os
import shutil
import logging

logger = logging.getLogger(__name__)


def move_files_to_finished(file_path, base_folder):
    """
    Moves a file to a 'finished' subdirectory within base_folder.
    Creates the 'finished' directory if it doesn't exist.
    
    Args:
        file_path: path to the file to move
        base_folder: parent folder where 'finished' subdir will be created
    """
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist, skipping move: {file_path}")
        return False

    finished_dir = os.path.join(base_folder, "finished")
    os.makedirs(finished_dir, exist_ok=True)

    dest_path = os.path.join(finished_dir, os.path.basename(file_path))
    try:
        shutil.move(file_path, dest_path)
        logger.info(f"Moved {os.path.basename(file_path)} to finished/")
        return True
    except Exception as e:
        logger.warning(f"Could not move {os.path.basename(file_path)}: {e}")
        return False
