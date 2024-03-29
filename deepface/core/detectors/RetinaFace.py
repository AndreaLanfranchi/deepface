from typing import Any, Dict, List, Optional

import numpy
from retinaface import RetinaFace as rf

from deepface.core.detector import Detector as DetectorBase
from deepface.core.exceptions import FaceNotFoundError
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace, Point, RangeInt

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
        min_confidence: float = float(0.0),
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)
        detected_faces: List[DetectedFace] = []
        img_height, img_width = img.shape[:2]

        faces: Dict[str, Any] = rf.detect_faces(img, model=self._detector, threshold=0.9)
        for key in faces.keys():
            item: Dict[str, Any] = faces[key]
            score = float(item["score"])
            if score < min_confidence:
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
                left_eye: Optional[List[float]] = landmarks.get("left_eye")
                right_eye: Optional[List[float]] = landmarks.get("right_eye")
                if left_eye and right_eye:
                    le_point = Point(int(left_eye[0]), int(left_eye[1]))
                    re_point = Point(int(right_eye[0]), int(right_eye[1]))
                    points = {"le": le_point, "re": re_point}

            detected_faces.append(
                DetectedFace(
                    confidence=score,
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
