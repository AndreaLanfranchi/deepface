import base64
import re
import os
import binascii
from typing import Union, Tuple

# 3rd party
import numpy
import cv2
import requests

def load_image(source: Union[str, numpy.ndarray]) -> Tuple[numpy.ndarray, str]:
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

    if isinstance(source, numpy.ndarray):
        return source, "numpy array"
    if not isinstance(source, str):
        raise TypeError(f"Unsupoorted source type {type(source)}")

    source = source.strip()
    if len(source) == 0:
        raise ValueError("Invalid source. Empty string.")

    http_pattern = re.compile(r"^http(s)?://.*", re.IGNORECASE)
    if http_pattern.match(source):
        return _load_image_from_web(url=source), source

    base64_pattern = re.compile(r"^data:image\/.*", re.IGNORECASE)
    if base64_pattern.match(source):
        return _load_base64(uri=source), "base64 encoded string"

    # TODO : this is still unsafe as there are many other ways to
    # express the source of image to be loaded.
    # for example, an "ftp://" link or a "sftp://" link
    # as a result bailing out to a filesystem load might be
    # an issue

    return _load_image_from_file(filename=source), source

def _load_image_from_web(url: str) -> numpy.ndarray:
    """
    Loading an image from web

    Args:

        url: link for the image

    Returns:
        img (numpy.ndarray): equivalent to pre-loaded image from opencv (BGR format)

    Raises:
        HTTPError: if the response status code is not 200
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image_array = numpy.asarray(bytearray(response.raw.read()), dtype=numpy.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def _load_base64(uri: str) -> numpy.ndarray:
    """Load image from base64 string.

    Args:
        uri: a base64 string.

    Returns:
        numpy array: the loaded image.

    Raises:
        ValueError: if the input is invalid.
    """

    split_data = uri.split(",")
    if len(split_data) != 2:
        raise ValueError("Invalid base64 input")
    pattern = re.compile(r"^data:image\/(jpeg|jpg|png)?(;base64)$", re.IGNORECASE)
    if not pattern.match(split_data[0]):
        raise ValueError("Invalid mime-type. Supported types are jpeg, jpg and png.")
    try:
        decoded = base64.b64decode(split_data[1], validate=True)
        nparr = numpy.frombuffer(buffer=decoded, dtype=numpy.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    except binascii.Error as ex:
        raise ValueError("Invalid base64 input") from ex

def _load_image_from_file(filename: str) -> numpy.ndarray:
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

def normalize_input(img: numpy.ndarray, normalization: str = "base") -> numpy.ndarray:
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
        raise NotImplementedError(f"Unimplemented normalization type : {normalization}")

    return img
