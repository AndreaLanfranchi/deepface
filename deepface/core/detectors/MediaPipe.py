from typing import Any, List, Optional

import math
import numpy

from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.core.types import BoxDimensions, InPictureFace, Point, RangeInt
from deepface.commons.logger import Logger
from deepface.core.exceptions import MissingOptionalDependency

try:
    from mediapipe.python.solutions.face_detection import FaceDetection
except ModuleNotFoundError:
    what: str = f"{__name__} requires `mediapipe` library."
    what += "You can install by 'pip install mediapipe' "
    raise MissingOptionalDependency(what) from None

logger = Logger.get_instance()


# MediaPipe detector (optional)
# see also: https://google.github.io/mediapipe/solutions/face_detection
class Detector(DetectorBase):

    _detector: FaceDetection

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        #   min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        #     detection to be considered successful (default 0.5). See details in
        #     https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
        #   model_selection: 0 or 1. 0 to select a short-range model that works
        #     best for faces within 2 meters from the camera, and 1 for a full-range
        #     model best for faces within 5 meters. (default 0) See details in
        #     https://solutions.mediapipe.dev/face_detection#model_selection.
        self._detector = FaceDetection(min_detection_confidence=0.7)

    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.0,
    ) -> List[InPictureFace]:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        results: List[InPictureFace] = []

        img_height, img_width = img.shape[:2]
        detection_result = self._detector.process(img)

        # Extract the bounding box, the landmarks and the confidence score
        for detection in detection_result.detections:
            if detection is None:
                continue

            (confidence,) = round(detection.score, 2)
            if confidence < min_confidence:
                continue

            bounding_box = detection.bounding_box
            x_range = RangeInt(
                bounding_box.origin_x, bounding_box.origin_x + bounding_box.width
            )
            y_range = RangeInt(
                bounding_box.origin_y, bounding_box.origin_y + bounding_box.height
            )
            if x_range.span <= 0 or y_range.span <= 0:
                continue  # Invalid detection

            if min_dims is not None:
                if min_dims.width > 0 and x_range.span < min_dims.width:
                    continue
                if min_dims.height > 0 and y_range.span < min_dims.height:
                    continue

            le_point = None
            re_point = None
            if detection.keypoints is not None and len(detection.keypoints) >= 2:
                # left eye and right eye 0 and 1
                # nose 2
                # mouth 3
                # right ear 4
                # left ear 5
                for i in range(2):
                    x = min(
                        math.floor(detection.keypoints[i].x * img_width), img_width - 1
                    )
                    y = min(
                        math.floor(detection.keypoints[i].y * img_height),
                        img_height - 1,
                    )
                    match i:
                        case 0:
                            le_point = Point(x, y)
                        case 1:
                            re_point = Point(x, y)
                        case _:
                            # Should not happen
                            raise IndexError("Index out of range")

                # Martian positions ?
                # TODO Decide whether to discard the face or to not include the eyes
                if not x_range.contains(le_point.x) or not y_range.contains(le_point.y):
                    le_point = None
                if not x_range.contains(re_point.x) or not y_range.contains(re_point.y):
                    re_point = None

            results.append(
                InPictureFace(
                    detector=self.name,
                    source=img,
                    y_range=y_range,
                    x_range=x_range,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=confidence,
                )
            )

        return results
