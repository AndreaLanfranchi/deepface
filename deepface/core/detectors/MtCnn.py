from typing import List, Optional

import cv2
import numpy
from mtcnn import MTCNN

from deepface.core.exceptions import FaceNotFoundError
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)
from deepface.core.detector import Detector as DetectorBase


# MtCnn detector
class Detector(DetectorBase):
    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._detector = MTCNN()

    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.0,
        raise_notfound: bool = False,
        detect_eyes: bool = True,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        results: List[DetectedFace] = []

        # mtcnn expects RGB but OpenCV read BGR
        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]

        detections = self._detector.detect_faces(img_rgb)

        if detections is not None:
            for current_detection in detections:

                confidence = float(current_detection["confidence"])
                if min_confidence is not None and confidence < min_confidence:
                    continue

                x, y, w, h = tuple(int(val) for val in current_detection["box"])
                x_range = RangeInt(x, min(x + w, img_width))
                y_range = RangeInt(y, min(y + h, img_height))
                if x_range.span <= 0 or y_range.span <= 0:
                    continue
                if isinstance(min_dims, BoxDimensions):
                    if min_dims.width > 0 and x_range.span < min_dims.width:
                        continue
                    if min_dims.height > 0 and y_range.span < min_dims.height:
                        continue

                bounding_box: BoundingBox = BoundingBox(
                    top_left=Point(x=x, y=y),
                    bottom_right=Point(x=x + w, y=y + h),
                )

                le_point = None
                re_point = None
                keypoints = current_detection.get("keypoints", None)
                if detect_eyes and keypoints is not None:
                    left_eye = keypoints["left_eye"]
                    right_eye = keypoints["right_eye"]
                    le_point = Point(x=int(left_eye[0]), y=int(left_eye[1]))
                    re_point = Point(x=int(right_eye[0]), y=int(right_eye[1]))
                    # Martian positions ?
                    # TODO Decide whether to discard the face or to not include the eyes
                    if le_point not in bounding_box or re_point not in bounding_box:
                        le_point = None
                        re_point = None
                    else:
                        if le_point < re_point:
                            le_point, re_point = re_point, le_point

                results.append(
                    DetectedFace(
                        bounding_box=bounding_box,
                        left_eye=le_point,
                        right_eye=re_point,
                        confidence=confidence,
                    )
                )

        if len(results) == 0 and raise_notfound == True:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            img=img,
            detections=results,
        )
