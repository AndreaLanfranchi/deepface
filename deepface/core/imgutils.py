from typing import List, Optional, Tuple, Union

import base64
import re
import os
import binascii
import requests

from PIL import Image
import numpy
import cv2

kKiB: int = 1024
KMiB: int = 1024 ** 2
kGiB: int = 1024 ** 3
kVALID_IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png", "webp"]

_kHTTP_PATTERN = re.compile(r"^http(s)?://.*", re.IGNORECASE)
_kBASE64_PATTERN = re.compile(r"^data:image\/.*", re.IGNORECASE)
_kBASE64_PATTERN_EXT = re.compile(
    r"^data:image\/(jpeg|jpg|png|webp)?(;base64)$", re.IGNORECASE
)

def is_valid_image(img: numpy.ndarray) -> bool:
    """
    Check if the image is valid

    Params:
    -------
        `img`: numpy.ndarray Image to check

    Returns:
    --------
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
        if img.shape[2] not in [1, 3]:  # grayscale or BGR/RGB
            return False
    else:
        if not numpy.issubdtype(img.dtype, numpy.uint8):
            return False

    return True


def is_valid_image_file(
    filename: str,
    check_ext: bool = False,
    max_size: int = 10 * KMiB,
) -> bool:
    """
    Check if the image file is valid

    Params:
    -------
        `filename`: str Image file to test
        `check_ext`: bool Check the file extension. This further checks the
            file extension to be one of the supported image types. 
            Default is False
        `max_size`: int Maximum file size in bytes. Default is 10 MiB

    Remarks:
    --------
        Regardless the extension check, the file content is also checked

    Returns:
    --------
        True if the image file is valid, False otherwise

    Raises:
    -------
        `TypeError` : If filename is not a string
        `IOError` : If the file cannot be opened or read
        `FileNotFoundError` : If the file does not exist
    """

    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    ret: bool = True
    filename = filename.strip()
    if len(filename) == 0 or not os.path.isfile(filename):
        raise FileNotFoundError(f"File [{filename}] does not exist")
    
    file_size: int = os.path.getsize(filename)
    if 0 == file_size or file_size > max_size:
        ret = False

    if ret and check_ext:
        _, ext = os.path.splitext(filename)
        ext = ext.strip(".").lower()
        if not ext in kVALID_IMAGE_TYPES:
            ret = False
    if not ret:
        return ret

    try:
        with Image.open(filename) as img:
            if img.format is None:
                ret = False
            elif img.format.lower() not in kVALID_IMAGE_TYPES:
                ret = False
    except Exception as ex:
        raise IOError(f"Error opening image file {filename}") from ex

    return ret


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


def get_all_image_files(
    directory: str,
    recurse: bool = True,
    check_ext: bool = False,
) -> List[str]:
    """
    Get all valid image files in a directory.

    Args:
    -----
        `directory`: the directory to search for image files. An empty string
            defaults to the current working directory (same as ".")
        `recurse`: whether to search recursively

    Returns:
    --------
        A list of valid image file names

    Raises:
    -------
        TypeError: if the directory argument is not a string
        `FileNotFoundError`: if the directory does not exist
    """

    if not isinstance(directory, str):
        raise TypeError("Directory must be a string")

    directory = directory.strip()
    if len(directory) == 0:
        directory = os.getcwd()

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    valid_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_image_file(file_path, check_ext=check_ext):
                valid_files.append(file_path)
        if not recurse:
            break

    return valid_files


def load_image(
    source: Union[str, numpy.ndarray],
    tag: Optional[str] = None,
) -> Tuple[numpy.ndarray, str]:
    """
    Load image from path, url, base64 or numpy array.

    Args:
    -----
        `source`: the origin of image data to be loaded

        `tag`: a tag to be returned with the image
        If none the returned tag will be determined as follows:
        - if the source is an Url, the tag will be the Url itself
        - if the source is a base64 string, the tag will be "base64"
        - if the source is a numpy array, the tag will be the type of the array


    Returns:
    --------
        A tuple of :\n
        `numpy.ndarray`: the loaded image in BGR format\n
        `str`: image tag\n

    Raises:
    -------
        `ValueError`: if the input is somewhat invalid
        `TypeError`: if the input is not a supported type
        `HTTPError`: if the input is a url and the response status code is not 200
        `FileNotFoundError`: if the input is a path and the file does not exist
    """

    if source is None or not isinstance(source, (str, numpy.ndarray)):
        what: str = "Invalid source. Expected [str | numpy.ndarray] "
        what += f"got {type(source)}"
        raise TypeError(what)
    if tag is not None and not isinstance(tag, str):
        what: str = 'Invalid "tag" type. Expected [str | None], '
        what += f"got {type(tag)}"
        raise TypeError(what)

    loaded_image: numpy.ndarray = numpy.array([])

    if isinstance(source, str):
        origin: str = str(source.strip())
        if len(origin) == 0:
            raise ValueError("Invalid source. Empty string.")
        if _kHTTP_PATTERN.match(origin):
            loaded_image = _load_image_from_web(url=origin)
            if tag is None:
                tag = "web"
        elif _kBASE64_PATTERN.match(origin):
            loaded_image = _load_base64(uri=origin)
            if tag is None:
                tag = "base64"
        else:
            # TODO : this is still unsafe as there are many other ways to
            # express the source of image to be loaded.
            # for example, an "ftp://" link or a "sftp://" link
            # as a result bailing out to a filesystem load might be
            # an issue
            loaded_image = _load_image_from_file(filename=origin)
            if tag is None:
                tag = origin

    if isinstance(source, numpy.ndarray):
        loaded_image = source
        if tag is None:
            tag = f"{type(source)}"

    if not is_valid_image(loaded_image):
        what: str = "Invalid image from "
        what += f"{tag}"
        raise ValueError("Invalid image from web")

    return (loaded_image, tag if tag is not None else "<unknown>")


def _load_image_from_web(url: str) -> numpy.ndarray:
    """
    Loading an image from web

    Args:
    -----
        `url`: link for the image

    Returns:
    --------
        `img` (numpy.ndarray): equivalent to pre-loaded image from opencv (BGR format)

    Raises:
    -------
        `HTTPError`: if the response status code is not 200
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
        `uri`: a base64 string.

    Returns:
    --------
        numpy array: the loaded image.

    Raises:
    -------
        `ValueError`: if the input is invalid.
    """

    # TODO : use regex capture groups to traverse the input string
    # in one pass only
    split_data = uri.split(",")
    if 2 != len(split_data):
        raise ValueError("Invalid base64 input")
    if not _kBASE64_PATTERN_EXT.match(split_data[0]):
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
        `filename`: full or relative path to the image file.

    Returns:
    --------
        numpy array: the loaded image.

    Raises:
    -------
        `FileNotFoundError`: if the file does not exist.
        `ValueError`: if the file is not a valid image file.
    """

    if not is_valid_image_file(filename):
        raise ValueError(f"{filename} is an invalid image file")

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
    mode = mode.lower().strip()
    if mode == "base":
        return img

    # @trevorgribble and @davedgd contributed this feature
    # restore input in scale of [0, 255] because it was normalized in scale of
    # [0, 1] in preprocess_face
    img *= 255

    if mode == "raw":
        pass  # return just restored pixels

    elif mode == "facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif mode == "facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif mode == "vggface":
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif mode == "vggface2":
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif mode == "arcface":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        raise NotImplementedError(f"Unimplemented normalization mode : {mode}")

    return img
