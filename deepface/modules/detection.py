# built-in dependencies
from typing import Any, Dict, List, Tuple, Union, Optional

# 3rd part dependencies
import numpy
import cv2
from PIL import Image

# project dependencies
from deepface.modules import preprocessing
from deepface.models.Detector import Detector
from deepface.detectors import DetectorWrapper
from deepface.commons import package_utils
from deepface.commons.logger import Logger

logger = Logger(module="deepface/modules/detection.py")

tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image


def detect_faces(
    source: Union[str, numpy.ndarray],
    detector: Union[str, Detector] = "opencv",
    target_size: Optional[Tuple[int, int]] = None,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    human_readable= False,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image

    Args:
    
        source (str or numpy.ndarray): If a string is provided, it is assumed the image have
        to be loaded/parsed from a file or base64 encoded string

        detector (str or Detector): If a string is provided, it is assumed the detector
        instance have to be lazily initialized.

        target_size (tuple): final shape of facial image. black pixels will be
        added to resize the image.

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage

        grayscale (boolean): Wether to convert the detected face image to grayscale.
        Default is False.

        human_readable (bool): Wether to make the image human readable. 3D RGB for human readable
        or 4D BGR for ML models. Default is False.

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (numpy.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values

        - "confidence" (float): The confidence score associated with the detected face.
    """

    results: List[Dict[str, Any]] = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img, _ = preprocessing.load_image(source)

    face_objs = DetectorWrapper.detect_faces(
        source=img,
        detector=detector,
        align=align,
        expand_percentage=expand_percentage,
    )

    for face_obj in face_objs:
        current_img = face_obj.img
        current_region = face_obj.facial_area

        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue

        if grayscale is True:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        # resize and padding
        if target_size is not None:
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]
            if grayscale is False:
                # Put the base image in the middle of the padded image
                current_img = numpy.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                current_img = numpy.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                    ),
                    "constant",
                )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

        # normalizing the image pixels
        # what this line doing? must?
        img_pixels = image.img_to_array(current_img)
        img_pixels = numpy.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]
        # discard expanded dimension
        if human_readable is True and len(img_pixels.shape) == 4:
            img_pixels = img_pixels[0]

        results.append(
            {
                "face": img_pixels[:, :, ::-1] if human_readable is True else img_pixels,
                "facial_area": {
                    "x": int(current_region.x),
                    "y": int(current_region.y),
                    "w": int(current_region.w),
                    "h": int(current_region.h),
                    "left_eye": current_region.left_eye,
                    "right_eye": current_region.right_eye,
                },
                "confidence": current_region.confidence,
            }
        )

    return results


def align_face(
    img: numpy.ndarray,
    left_eye: Union[list, tuple],
    right_eye: Union[list, tuple],
) -> Tuple[numpy.ndarray, float]:
    """
    Align a given image horizantally with respect to their left and right eye locations
    Args:
        img (numpy.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the you
        right_eye(list or tuple): coordinates of right eye with respect to the you
    Returns:
        img (numpy.ndarray): aligned facial image
    """
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img, 0

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(numpy.degrees(numpy.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    img = numpy.array(Image.fromarray(img).rotate(angle))
    return img, angle
