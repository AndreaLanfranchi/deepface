from typing import List, Optional

import math
import numpy

from deepface.core.detector import Detector as DetectorBase
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)
from deepface.commons.logger import Logger
from deepface.core.exceptions import FaceNotFound, MissingOptionalDependency

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
        raise_notfound: bool = False,
    ) -> DetectorBase.Outcome:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        detected_faces: List[DetectedFace] = []

        img_height, img_width = img.shape[:2]
        detection_result = self._detector.process(img)

        # Extract the bounding box, the landmarks and the confidence score
        for detection in detection_result.detections:
            if detection is None:
                continue

            (confidence,) = round(detection.score, 2)
            if min_confidence is not None and confidence < min_confidence:
                continue

            detection_box = detection.bounding_box
            x_range = RangeInt(
                detection_box.origin_x,
                min(detection_box.origin_x + detection_box.width, img_width),
            )
            y_range = RangeInt(
                detection_box.origin_y,
                min(detection_box.origin_y + detection_box.height, img_height),
            )
            if x_range.span <= 0 or y_range.span <= 0:
                continue  # Invalid detection

            if min_dims is not None:
                if min_dims.width > 0 and x_range.span < min_dims.width:
                    continue
                if min_dims.height > 0 and y_range.span < min_dims.height:
                    continue

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

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
                if le_point not in bounding_box or re_point not in bounding_box:
                    le_point = None
                    re_point = None

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=float(confidence),
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFound("No face detected. Check the input image.")

        return DetectorBase.Outcome(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
