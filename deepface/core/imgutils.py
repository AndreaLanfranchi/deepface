from typing import List, Tuple, Union

import base64
import re
import os
import binascii
import requests

import numpy
import cv2


def is_valid_image(img: numpy.ndarray) -> bool:
    """
    Check if the image is valid

    Params:
    -------
    image: numpy.ndarray
        Image to check

    Returns:
    --------
    bool
        True if the image is valid, False otherwise

    Raises:
    -------
        None
    """

    if not isinstance(img, numpy.ndarray):
        return False

    if img.ndim not in [2, 3]:
        return False

    if img.shape[0] == 0 or img.shape[1] == 0:
        return False

    if img.ndim == 3:
        if img.shape[2] not in [1, 3]:  # grayscale or BGR
            return False
    else:
        if not numpy.issubdtype(img.dtype, numpy.uint8):
            return False

    return True


def is_valid_image_file(filename: str) -> bool:
    """
    Check if the image file is valid

    Params:
    -------
    filename: str
        Image file to check

    Returns:
    --------
    bool
        True if the image file is valid, False otherwise

    Raises:
    -------
        FileNotFoundError
    """

    if not isinstance(filename, str):
        return False

    # TODO: check if the file is a valid image file
    # using the magic number of the file
    # as the file extension is not a reliable way to check
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if not ext in [".jpg", ".jpeg", ".png"]:
        return False

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    if os.path.getsize(filename) == 0:
        raise ValueError(f"File {filename} is empty")

    return True


def is_grayscale_image(img: numpy.ndarray) -> bool:
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
        If the image is not a valid numpy array
    """

    if not is_valid_image(img):
        raise ValueError("Image must be a valid numpy array for a single image")

    if img.ndim == 2:
        return True

    if img.shape[2] == 1:
        return True

    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all():
        return True

    return False


# Pseudo-constants
_http_pattern = re.compile(r"^http(s)?://.*", re.IGNORECASE)
_base64_pattern = re.compile(r"^data:image\/.*", re.IGNORECASE)
_base64_pattern_ext = re.compile(
    r"^data:image\/(jpeg|jpg|png)?(;base64)$", re.IGNORECASE
)


def get_all_valid_files(directory: str, recurse: bool = True) -> List[str]:
    """
    Get all valid image files in a directory.

    Args:
    -----
        directory: the directory to search for image files
        recurse: whether to search recursively or not

    Returns:
    --------
        A list of valid image files

    Raises:
    -------
        FileNotFoundError: if the directory does not exist
    """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    valid_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_image_file(file_path):
                valid_files.append(file_path)

        if not recurse:
            break

    return valid_files


def load_image(source: Union[str, numpy.ndarray]) -> Tuple[numpy.ndarray, str]:
    """
    Load image from path, url, base64 or numpy array.

    Args:
    -----
        source: the origin of image data to be loaded

    Returns:
    --------
        A tuple of :\n
        image (numpy array): the loaded image in BGR format\n
        name (str): image name itself\n

    Raises:
    -------
        ValueError: if the input is somewhat invalid
        TypeError: if the input is not a supported type
        HTTPError: if the input is a url and the response status code is not 200
        FileNotFoundError: if the input is a path and the file does not exist
    """

    if source is None:
        raise TypeError("Invalid source. None type.")

    loaded_image: numpy.ndarray = numpy.array([])
    tag: str = ""

    if isinstance(source, str):
        origin: str = str(source.strip())
        if len(origin) == 0:
            raise ValueError("Invalid source. Empty string.")
        if _http_pattern.match(origin):
            loaded_image = _load_image_from_web(url=origin)
            tag = "web"
        elif _base64_pattern.match(origin):
            loaded_image = _load_base64(uri=origin)
            tag = "base64"
        else:
            # TODO : this is still unsafe as there are many other ways to
            # express the source of image to be loaded.
            # for example, an "ftp://" link or a "sftp://" link
            # as a result bailing out to a filesystem load might be
            # an issue
            loaded_image = _load_image_from_file(filename=origin)
            tag = f"file {origin}"

        if not is_valid_image(loaded_image):
            what: str = "Invalid image from "
            what += f"{tag}"
            raise ValueError("Invalid image from web")
        return (loaded_image, tag)

    if isinstance(source, numpy.ndarray):
        if not is_valid_image(source):
            raise ValueError("Invalid image")
        return (source, f"{type(source)}")

    raise TypeError("Invalid source type. Expected str or numpy.ndarray.")


def _load_image_from_web(url: str) -> numpy.ndarray:
    """
    Loading an image from web

    Args:
    -----
        url: link for the image

    Returns:
    --------
        img (numpy.ndarray): equivalent to pre-loaded image from opencv (BGR format)

    Raises:
    -------
        HTTPError: if the response status code is not 200
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image_array = numpy.asarray(bytearray(response.raw.read()), dtype=numpy.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)
    return image


def _load_base64(uri: str) -> numpy.ndarray:
    """
    Load image from base64 string.

    Args:
    -----
        uri: a base64 string.

    Returns:
    --------
        numpy array: the loaded image.

    Raises:
    -------
        ValueError: if the input is invalid.
    """

    # TODO : use regex capture groups to traverse the input string
    # in one pass only
    split_data = uri.split(",")
    if len(split_data) != 2:
        raise ValueError("Invalid base64 input")
    if not _base64_pattern_ext.match(split_data[0]):
        raise ValueError("Invalid mime-type. Supported types are jpeg, jpg and png.")
    try:
        decoded = base64.b64decode(split_data[1], validate=True)
        nparr = numpy.frombuffer(buffer=decoded, dtype=numpy.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    except binascii.Error as ex:
        raise ValueError("Invalid base64 input") from ex


def _load_image_from_file(filename: str) -> numpy.ndarray:
    """
    Load image from file.

    Args:
    -----
        filename: full or relative path to the image file.

    Returns:
    --------
        numpy array: the loaded image.

    Raises:
    -------
        FileNotFoundError: if the file does not exist.

    """

    if not is_valid_image_file(filename):
        raise ValueError("Invalid image file")

    return cv2.imread(filename)


def normalize_input(img: numpy.ndarray, mode: str = "base") -> numpy.ndarray:
    """
    Normalize input image.

    Args:
    -----
        img (numpy array): the input image.
        mode (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
    --------
        numpy array: the normalized image.

    Raises:
    -------
        NotImplementedError: if the normalization technique is not implemented.
    """

    if img.ndim == 4:
        img = img[0]

    if not is_valid_image(img):
        what: str = "Invalid input image type. Expected numpy.ndarray, "
        what += f"got {type(img)}"
        raise TypeError(what)

    if not isinstance(mode, str):
        what: str = 'Invalid "mode" type. Expected str, '
        what += f"got {type(mode)}"
        raise TypeError(what)

    # issue 131 declares that some normalization techniques improves the accuracy

    if mode == "base":
        return img

    # @trevorgribble and @davedgd contributed this feature
    # restore input in scale of [0, 255] because it was normalized in scale of
    # [0, 1] in preprocess_face
    img *= 255

    if mode == "raw":
        pass  # return just restored pixels

    elif mode == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif mode == "Facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif mode == "VGGFace":
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif mode == "VGGFace2":
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif mode == "ArcFace":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        raise NotImplementedError(f"Unimplemented normalization mode : {mode}")

    return img
