from typing import List, Optional

import numpy
from mtcnn import MTCNN

from deepface.core.exceptions import FaceNotFound
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
    ) -> DetectorBase.Outcome:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        detected_faces: List[DetectedFace] = []

        # mtcnn expects RGB but OpenCV read BGR
        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = img[:, :, ::-1]
        img_height, img_width = img.shape[:2]

        detections = self._detector.detect_faces(img_rgb)

        if detections is not None:
            for current_detection in detections:

                confidence = current_detection["confidence"]
                if min_confidence is not None and confidence < min_confidence:
                    continue

                x, y, w, h = current_detection["box"]
                x_range = RangeInt(
                    x,
                    min(x + w, img_width),
                )
                y_range = RangeInt(
                    y,
                    min(y + h, img_height),
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
                if "keypoints" in current_detection.keys():
                    left_eye = current_detection["keypoints"]["left_eye"]
                    right_eye = current_detection["keypoints"]["right_eye"]
                    le_point = Point(x=left_eye[0], y=left_eye[1])
                    re_point = Point(x=right_eye[0], y=right_eye[1])
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
            detector=self._name,
            source=img,
            detected_faces=detected_faces,
        )
