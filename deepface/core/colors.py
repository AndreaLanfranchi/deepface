from typing import Tuple

import numpy

KBGR_COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
KBGR_COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)
KBGR_COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)
KBGR_COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
KBGR_COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
KBGR_COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)
KBGR_COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)

KBGR_COLOR_LEFT_EYE: Tuple[int, int, int] = KBGR_COLOR_RED
KBGR_COLOR_RIGHT_EYE: Tuple[int, int, int] = KBGR_COLOR_BLUE
KBGR_COLOR_NOSE: Tuple[int, int, int] = KBGR_COLOR_GREEN
KBGR_COLOR_LEFT_MOUTH: Tuple[int, int, int] = KBGR_COLOR_RED
KBGR_COLOR_RIGHT_MOUTH: Tuple[int, int, int] = KBGR_COLOR_BLUE
KBGR_COLOR_CENTER_MOUTH: Tuple[int, int, int] = KBGR_COLOR_GREEN
KBGR_COLOR_BOUNDING_BOX: Tuple[int, int, int] = KBGR_COLOR_CYAN


def is_gray_scale(image: numpy.ndarray) -> bool:
    """
    Check if the image is in grayscale

    Params:
    -------
    image: numpy.ndarray
        Image to check

    Returns:
    --------
    bool
        True if the image is in grayscale, False otherwise

    Raises:
    -------
    ValueError
        If the image is not a numpy array
    """

    if not isinstance(image, numpy.ndarray) or image.ndim < 2 or image.ndim > 3:
        raise ValueError("Image must be a valid numpy array for a single image")

    if image.ndim == 2:
        return True

    if image.shape[2] == 1:
        return True

    if (image[:, :, 0] == image[:, :, 1]).all() and (
        image[:, :, 1] == image[:, :, 2]
    ).all():
        return True

    return False
