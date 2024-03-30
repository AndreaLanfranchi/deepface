from typing import Any, Dict, List, Optional

import numpy
from retinaface import RetinaFace as rf

from deepface.core.detector import Detector as DetectorBase
from deepface.core.exceptions import FaceNotFoundError
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)

from deepface.commons.logger import Logger

logger = Logger.get_instance()


# RetinaFace detector
class Detector(DetectorBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._detector = rf.build_model()

    def process(
        self,
        img: numpy.ndarray,
        tag: Optional[str] = None,
        min_dims: BoxDimensions = BoxDimensions(0, 0),
        min_confidence: float = float(0.95),
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)
        detected_faces: List[DetectedFace] = []
        img_height, img_width = img.shape[:2]

        faces: Dict[str, Any] = rf.detect_faces(
            img, model=self._detector, threshold=min_confidence
        )
        for key in faces.keys():
            item: Dict[str, Any] = faces[key]
            confidence = float(item["score"])
            if confidence < min_confidence:
                continue

            x1, y1, x2, y2 = (int(val) for val in item["facial_area"][:4])
            x_range = RangeInt(x1, min(x2, img_width))
            y_range = RangeInt(y1, min(y2, img_height))
            if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                continue  # Invalid or empty detection

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            points: Optional[Dict[str, Optional[Point]]] = None
            if key_points:
                landmarks: Dict[str, List[float]] = item["landmarks"]
                points = {}
                left_xy: Optional[List[float]] = landmarks.get("left_eye")
                right_xy: Optional[List[float]] = landmarks.get("right_eye")
                if left_xy and right_xy:
                    left_point = Point(int(round(left_xy[0])), int(round(left_xy[1])))
                    right_point = Point(
                        int(round(right_xy[0])), int(round(right_xy[1]))
                    )
                    points.update({"lec": left_point, "rec": right_point})

                left_xy: Optional[List[float]] = landmarks.get("nose")
                if left_xy:
                    nt_point = Point(int(left_xy[0]), int(left_xy[1]))
                    points.update({"nt": nt_point})

                left_xy: Optional[List[float]] = landmarks.get("mouth_left")
                right_xy: Optional[List[float]] = landmarks.get("mouth_right")
                if left_xy and right_xy:
                    left_point = Point(int(round(left_xy[0])), int(round(left_xy[1])))
                    right_point = Point(
                        int(round(right_xy[0])), int(round(right_xy[1]))
                    )
                    center_point = Point(
                        x=(left_point.x + right_point.x) // 2,
                        y=(left_point.y + right_point.y) // 2,
                    )
                    points.update(
                        {"mlc": left_point, "mrc": right_point, "mc": center_point}
                    )

            try:
                # This might raise an exception if the values are out of bounds
                detected_faces.append(
                    DetectedFace(
                        confidence=confidence,
                        bounding_box=bounding_box,
                        key_points=points,
                    )
                )
            except Exception as e:
                logger.debug(f"Error: {e}")

        if 0 == len(detected_faces) and raise_notfound:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            img=img,
            tag=tag,
            detections=detected_faces,
        )
