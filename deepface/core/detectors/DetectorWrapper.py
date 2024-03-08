# built-in dependencies
from typing import List, Tuple, Union

# 3rd party dependencies
import numpy

# project dependencies
from deepface.modules import detection, preprocessing
from deepface.core.detector import Detector, DetectedFace, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger.get_instance()


def detect_faces(
    source: Union[str, numpy.ndarray],
    detector: Union[str, Detector] = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
) -> List[DetectedFace]:
    """

    Tries to detect face(s) from a provided image.

    Args:

        source (str or numpy.ndarray): If a string is provided, it is assumed the image have
        to be loaded/parsed from a file or base64 encoded string

        detector (str or Detector): If a string is provided, it is assumed the detector
        instance have to be lazily initialized.

        align (bool): wether to perform alignment after detection. Default is True.

        expand_percentage (int): expand detected facial area with a percentage. Default is 0.
        Negative values are clamped to 0.

    Returns:
        results (List[DetectedFace]): A list of DetectedFace objects
            where each object contains:

        - img (numpy.ndarray): The detected face as a NumPy array.

        - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h
    """

    # Validation
    if isinstance(detector, str):
        detector = Detector.instance(detector)
    if isinstance(source, str):
        source, _ = preprocessing.load_image(source)

    expand_percentage = max(0, expand_percentage)  # clamp to 0

    # If the image is too small, return an empty list
    # TODO: Add a warning message here ?
    # TODO: Maybe set a minimum size for the image globally ?
    if source.shape[0] < 32 or source.shape[1] < 32:
        return []

    # find facial areas of given image
    facial_areas = detector.process(img=source)

    results = []
    for facial_area in facial_areas:
        x = facial_area.x
        y = facial_area.y
        w = facial_area.w
        h = facial_area.h
        left_eye = facial_area.left_eye
        right_eye = facial_area.right_eye
        confidence = facial_area.confidence

        if expand_percentage > 0:
            # Expand the facial region height and width by the provided percentage
            # ensuring that the expanded region stays within img.shape limits
            expanded_w = w + int(w * expand_percentage / 100)
            expanded_h = h + int(h * expand_percentage / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(source.shape[1] - x, expanded_w)
            h = min(source.shape[0] - y, expanded_h)

        # extract detected face unaligned
        detected_face = source[int(y) : int(y + h), int(x) : int(x + w)]

        # align original image, then find projection of detected face area after alignment
        if align is True:  # and left_eye is not None and right_eye is not None:
            aligned_img, angle = detection.align_face(
                img=source, left_eye=left_eye, right_eye=right_eye
            )
            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
                facial_area=(x, y, x + w, y + h),
                angle=angle,
                size=(source.shape[0], source.shape[1]),
            )
            detected_face = aligned_img[
                int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
            ]

        result = DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x,
                y=y,
                h=h,
                w=w,
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
            ),
        )
        results.append(result)
    return results


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.
    Inspried from the work of @UmutDeniz26 - github.com/serengil/retinaface/pull/80

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
            x2 is equal to x1 + w1, and y2 is equal to y1 + h1
        angle (float): Angle of rotation in degrees. Its sign determines the direction of rotation.
                       Note that angles > 360 degrees are normalized to the range [0, 360).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_coordinates (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) or (x1, y1, x1+w1, y1+h1) of the rotated facial area.
    """

    # Normalize the witdh of the angle so we don't have to
    # worry about rotations greater than 360 degrees.
    # We workaround the quirky behavior of the modulo operator
    # for negative angle values.
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area

    # Angle in radians
    angle = angle * numpy.pi / 180

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - size[1] / 2
    y = (facial_area[1] + facial_area[3]) / 2 - size[0] / 2

    # Rotate the facial area
    x_new = x * numpy.cos(angle) + y * direction * numpy.sin(angle)
    y_new = -x * direction * numpy.sin(angle) + y * numpy.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + size[1] / 2
    y_new = y_new + size[0] / 2

    # Calculate the new facial area
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    return (int(x1), int(y1), int(x2), int(y2))
