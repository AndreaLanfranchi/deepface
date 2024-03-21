from typing import Any, Dict, List, Optional

import numpy
from retinaface import RetinaFace as rf

from deepface.core.detector import Detector as DetectorBase
from deepface.core.exceptions import FaceNotFound
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace, Point, RangeInt

# RetinaFace detector
class Detector(DetectorBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._detector = rf.build_model()

    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.0,
        raise_notfound: bool = False,
    ) -> DetectorBase.Outcome:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        img_height, img_width = img.shape[:2]
        detected_faces: List[DetectedFace] = []

        faces: Dict[str, Any] = rf.detect_faces(img, model=self._detector, threshold=0.9)
        for key in faces.keys():
            item: Dict[str, Any] = faces[key]
            score = float(item["score"])
            if min_confidence is not None and score < min_confidence:
                continue

            x1, y1, x2, y2 = (
                int(item["facial_area"][0]),
                int(item["facial_area"][1]),
                int(item["facial_area"][2]),
                int(item["facial_area"][3]),
            )
            x_range = RangeInt(min(0, x1), min(x2, img_width))
            y_range = RangeInt(min(0, y1), min(y2, img_height))
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
            landmarks: Dict[str, List[float]] = item["landmarks"]
            if landmarks.get("left_eye") and landmarks.get("right_eye"):
                le_point = Point(
                    int(landmarks["left_eye"][0]), int(landmarks["left_eye"][1])
                )
                re_point = Point(
                    int(landmarks["right_eye"][0]), int(landmarks["right_eye"][1])
                )
                if le_point not in bounding_box or re_point not in bounding_box:
                    le_point = None
                    re_point = None

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=score,
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFound("No face detected. Check the input image.")

        return DetectorBase.Outcome(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
