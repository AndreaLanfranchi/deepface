from typing import Dict, List, Optional

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
        tag: Optional[str] = None,
        min_dims: BoxDimensions = BoxDimensions(0, 0),
        min_confidence: float = float(0.0),
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)
        detected_faces: List[DetectedFace] = []

        # mtcnn expects RGB but OpenCV read BGR
        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]

        detections = self._detector.detect_faces(img_rgb)

        if detections is not None:
            for current_detection in detections:

                confidence = float(current_detection["confidence"])
                if confidence < min_confidence:
                    continue

                x, y, w, h = tuple(int(val) for val in current_detection["box"])
                x_range = RangeInt(x, min(x + w, img_width))
                y_range = RangeInt(y, min(y + h, img_height))
                if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                    continue  # Invalid or empty detection

                bounding_box: BoundingBox = BoundingBox(
                    top_left=Point(x=x, y=y),
                    bottom_right=Point(x=x + w, y=y + h),
                )

                points: Optional[Dict[str, Optional[Point]]] = None
                keypoints = current_detection.get("keypoints", None)
                if key_points and keypoints is not None:

                    # TODO: Remove this print
                    print(f"keypoints: {keypoints}")

                    left_eye = keypoints["left_eye"]
                    right_eye = keypoints["right_eye"]
                    le_point = Point(x=int(left_eye[0]), y=int(left_eye[1]))
                    re_point = Point(x=int(right_eye[0]), y=int(right_eye[1]))
                    points = {"le": le_point, "re": re_point}

                detected_faces.append(
                    DetectedFace(
                        confidence=confidence,
                        bounding_box=bounding_box,
                        key_points=points,
                    )
                )

        if 0 == len(detected_faces) and raise_notfound:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            img=img,
            tag=tag,
            detections=detected_faces,
        )
