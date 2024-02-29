import os
from pathlib import Path
from deepface.commons.logger import Logger

logger = Logger(module="deepface/commons/folder_utils.py")

def get_deepface_home() -> str:
    """
    Get the home directory for storing model weights
    It gets the home directory from the environment variable DEEPFACE_HOME
    or the user's home directory when the former is not set.

    Returns:
        str: the full path to home directory.

    Raise:
        ValueError: if the home directory does not exist.
    """
    home_dir = str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))
    if not os.path.exists(home_dir):
        raise ValueError(f"Directory {home_dir} does not exist")
    home_dir_data = os.path.join(home_dir, ".deepface", "weights")
    if not os.path.exists(home_dir_data):
        os.makedirs(home_dir_data, exist_ok=True) # this is recursive
    return home_dir
