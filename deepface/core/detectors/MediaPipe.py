from typing import Any, List
import cv2

import numpy

from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons.logger import Logger
from deepface.core.exceptions import MissingOptionalDependency

try:
    import mediapipe as mp
except ModuleNotFoundError:
    what: str = f"{__name__} requires `mediapipe` library."
    what += "You can install by 'pip install mediapipe' "
    raise MissingOptionalDependency(what) from None

logger = Logger.get_instance()

# MediaPipe detector (optional)
# see also: https://google.github.io/mediapipe/solutions/face_detection
class Detector(DetectorBase):

    _detector: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        mp_face_detection = mp.solutions.face_detection
        self._detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.7
        )

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:

        results = []
        if len(img.shape) < 3 or img.shape[2] != 3:
            logger.debug("Converting image to RGB")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        img_width = img.shape[1]
        img_height = img.shape[0]
        outcome = self._detector.process(img)

        # If no face has been detected, return an empty list
        if outcome.detections is None:
            return results

        # Extract the bounding box, the landmarks and the confidence score
        for current_detection in outcome.detections:
            (confidence,) = current_detection.score

            bounding_box = current_detection.location_data.relative_bounding_box
            landmarks = current_detection.location_data.relative_keypoints

            x = int(bounding_box.xmin * img_width)
            w = int(bounding_box.width * img_width)
            y = int(bounding_box.ymin * img_height)
            h = int(bounding_box.height * img_height)

            left_eye = (
                int(landmarks[0].x * img_width),
                int(landmarks[0].y * img_height),
            )
            right_eye = (
                int(landmarks[1].x * img_width),
                int(landmarks[1].y * img_height),
            )
            # nose = (int(landmarks[2].x * img_width), int(landmarks[2].y * img_height))
            # mouth = (int(landmarks[3].x * img_width), int(landmarks[3].y * img_height))
            # right_ear = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
            # left_ear = (int(landmarks[5].x * img_width), int(landmarks[5].y * img_height))

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            results.append(facial_area)

        return results
