import os
import binascii
from typing import Any, Tuple
import base64
from pathlib import Path

# 3rd party
import numpy as np
import cv2
import requests

def load_image(source: Any) -> Tuple[np.ndarray, str]:
    """
    Load image from path, url, base64 or numpy array.

    Args:
        source: the origin of image data to be loaded

    Returns:
        A tuple of :\n
        image (numpy array): the loaded image in BGR format\n
        name (str): image name itself\n

    Raises:
        ValueError: if the input is somewhat invalid
        TypeError: if the input is not a supported type
        HTTPError: if the input is a url and the response status code is not 200
        FileNotFoundError: if the input is a path and the file does not exist
    """

    if source is None:
        raise ValueError("Invalid source. Cannot be None.")
    if isinstance(source, np.ndarray):
        return source, "numpy array"
    if isinstance(source, Path):
        source = str(source)
    elif not isinstance(source, str):
        raise TypeError(f"Unsupoorted source type {type(source)}")

    if len(source.replace(" ", "")) == 0:
        raise ValueError("Invalid source. Empty string.")
    if source.lower().startswith("data:image/"):
        return __load_base64(uri=source), "base64 encoded string"
    if source.lower().startswith("http"):
        return __load_image_from_web(url=source), source

    return __load_image_from_file(filename=source), source

def __load_image_from_web(url: str) -> np.ndarray:
    """
    Loading an image from web
    Args:
        url: link for the image
    Returns:
        img (np.ndarray): equivalent to pre-loaded image from opencv (BGR format)
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def __load_base64(uri: str) -> np.ndarray:
    """Load image from base64 string.

    Args:
        uri: a base64 string.

    Returns:
        numpy array: the loaded image.
    """
    encoded_data = uri.split(",")[1]
    try:
        decoded = base64.b64decode(encoded_data, validate=True)
        nparr = np.frombuffer(buffer=decoded, dtype=np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except binascii.Error as ex:
        raise ValueError("Invalid base64 input") from ex

def __load_image_from_file(filename: str) -> np.ndarray:
    """Load image from file.

    Args:
        filename: full or relative path to the image file.

    Returns:
        numpy array: the loaded image.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if not ext in [".jpg", ".jpeg", ".png"]:
        raise ValueError(f"Unsupported file type {ext}")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
    if os.path.getsize(filename) == 0:
        raise ValueError(f"File {filename} is empty")

    return cv2.imread(filename)

def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image.

    Args:
        img (numpy array): the input image.
        normalization (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
        numpy array: the normalized image.

    Raises:
        NotImplementedError: if the normalization technique is not implemented.
    """

    # issue 131 declares that some normalization techniques improves the accuracy

    if normalization == "base":
        return img

    # @trevorgribble and @davedgd contributed this feature
    # restore input in scale of [0, 255] because it was normalized in scale of
    # [0, 1] in preprocess_face
    img *= 255

    if normalization == "raw":
        pass  # return just restored pixels

    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif normalization == "Facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif normalization == "VGGFace":
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif normalization == "VGGFace2":
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif normalization == "ArcFace":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        raise NotImplementedError(f"Unimplemented normalization type - {normalization}")

    return img
