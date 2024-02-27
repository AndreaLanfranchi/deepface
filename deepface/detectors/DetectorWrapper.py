import math
from typing import Any, List, Tuple
import numpy as np
from deepface.modules import detection
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.detectors import (
    FastMtCnn,
    MediaPipe,
    MtCnn,
    OpenCv,
    Dlib,
    RetinaFace,
    Ssd,
    Yolo,
    YuNet,
)
from deepface.commons.logger import Logger

logger = Logger(module="deepface/detectors/DetectorWrapper.py")


def build_model(detector_backend: str) -> Any:
    """
    Build a face detector model
    Args:
        detector_backend (str): backend detector name
    Returns:
        built detector (Any)
    """

    global built_face_detector_objs   # singleton design pattern
    if not "built_face_detector_objs" in globals():
        built_face_detector_objs = {}

    global avaliable_face_detector_objs
    if not "avaliable_face_detector_objs" in globals():
        avaliable_face_detector_objs = {
            "opencv": OpenCv.OpenCvClient,
            "mtcnn": MtCnn.MtCnnClient,
            "ssd": Ssd.SsdClient,
            "dlib": Dlib.DlibClient,
            "retinaface": RetinaFace.RetinaFaceClient,
            "mediapipe": MediaPipe.MediaPipeClient,
            "yolov8": Yolo.YoloClient,
            "yunet": YuNet.YuNetClient,
            "fastmtcnn": FastMtCnn.FastMtCnnClient,
        }

    if detector_backend not in avaliable_face_detector_objs.keys():
        raise KeyError(f"Invalid detector_backend passed - {detector_backend}")

    if detector_backend not in built_face_detector_objs.keys():
        detector_obj = avaliable_face_detector_objs[detector_backend]()
        built_face_detector_objs[detector_backend] = detector_obj

    return built_face_detector_objs[detector_backend]


def detect_faces(
    img: np.ndarray,
    detector_backend: str,
    align: bool = True,
    expand_percentage: int = 0
) -> List[DetectedFace]:
    """
    Detect face(s) from a given image
    Args:
        detector_backend (str): detector name

        img (np.ndarray): pre-loaded image

        align (bool): enable or disable alignment after detection

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

    Returns:
        results (List[DetectedFace]): A list of DetectedFace objects
            where each object contains:

        - img (np.ndarray): The detected face as a NumPy array.

        - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h

        - confidence (float): The confidence score associated with the detected face.
    """
    face_detector: Detector = build_model(detector_backend)

    # validate expand percentage score
    if expand_percentage < 0:
        logger.warn(
            f"Expand percentage cannot be negative but you set it to {expand_percentage}."
            "Overwritten it to 0."
        )
        expand_percentage = 0

    # find facial areas of given image
    facial_areas = face_detector.detect_faces(img=img)

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
            # Uncomment this if you want to :
            # Expand the facial area to be extracted and recompute the height and width
            # keeping the same aspect ratio and ensuring that the expanded area stays
            # within img.shape limits

            # current_area = w * h
            # expanded_area = current_area + int((current_area * expand_percentage) / 100)
            # scale_factor = math.sqrt(expanded_area / current_area)
            # expanded_w = int(w * scale_factor)
            # expanded_h = int(h * scale_factor)

            # Or uncomment this if you want to :
            # Expand the facial region height and width by the provided percentage
            # ensuring that the expanded region stays within img.shape limits
            expanded_w = int(w * expand_percentage / 100)
            expanded_h = int(h * expand_percentage / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)
      

        # extract detected face unaligned
        detected_face = img[y : y + h, x : x + w]

        # align original image, then find projection of detected face area after alignment
        if align is True:  # and left_eye is not None and right_eye is not None:
            aligned_img, angle = detection.align_face(
                img=img, left_eye=left_eye, right_eye=right_eye
            )
            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
                facial_area=(x, y, x + w, y + h), 
                angle=angle, 
                direction=1, 
                size=(img.shape[0], img.shape[1])
            )
            detected_face = aligned_img[int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)]

        result = DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x, y=y, h=h, w=w, confidence=confidence, left_eye=left_eye, right_eye=right_eye
            ),
            confidence=confidence,
        )
        results.append(result)
    return results


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.
    Inspried from the work of @UmutDeniz26 - github.com/serengil/retinaface/pull/80

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
            x2 is equal to x1 + w1, and y2 is equal to y1 + h1
        angle (float): Angle of rotation in degrees.
        direction (int): Direction of rotation (-1 for clockwise, 1 for counterclockwise).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_coordinates (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) or (x1, y1, x1+w1, y1+h1) of the rotated facial area.
    """
    # Angle in radians
    angle = angle * np.pi / 180

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - size[1] / 2
    y = (facial_area[1] + facial_area[3]) / 2 - size[0] / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + size[1] / 2
    y_new = y_new + size[0] / 2

    # Calculate the new facial area
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    return (int(x1), int(y1), int(x2), int(y2))
