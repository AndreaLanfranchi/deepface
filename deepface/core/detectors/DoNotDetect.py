from typing import List, Optional

import numpy

from deepface.core.detector import Detector as DetectorBase
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace, Point


class Detector(DetectorBase):
    """
    This class is used to skip face detection. It is used when the user
    wants to use a pre-detected face.
    """

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])

    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.0,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        img_height, img_width = img.shape[:2]
        detected_faces: List[DetectedFace] = []

        bounding_box: BoundingBox = BoundingBox(
            top_left=Point(x=int(0), y=int(0)),
            bottom_right=Point(x=int(img_width), y=int(img_height)),
        )

        detected_faces.append(
            DetectedFace(
                bounding_box=bounding_box,
                confidence=float(0),
            )
        )

        return DetectorBase.Results(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
