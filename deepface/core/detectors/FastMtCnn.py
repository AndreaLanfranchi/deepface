from typing import List, Optional

import cv2
import numpy

from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)
from deepface.core.detector import Detector as DetectorBase
from deepface.core.exceptions import FaceNotFound, MissingOptionalDependency

try:
    from facenet_pytorch import MTCNN as fast_mtcnn
except ModuleNotFoundError:
    what: str = f"{__name__} requires `facenet-pytorch` library."
    what += "You can install by 'pip install facenet-pytorch' "
    raise MissingOptionalDependency(what) from None


# FastMtCnn detector (optional)
# see also:
# https://github.com/timesler/facenet-pytorch
# https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
class Detector(DetectorBase):

    _detector: fast_mtcnn

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        # TODO: Use CUDA if available
        self._detector = fast_mtcnn(device="cpu")

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

        # TODO: check image has less than 4 channels

        # mtcnn expects RGB but OpenCV read BGR
        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, probs, points = self._detector.detect(img_rgb, landmarks=True)

        for box, prob, point in zip(boxes, probs, points):

            if box is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue  # No detection or tampered data

            if min_confidence is not None and prob < min_confidence:
                continue  # Confidence too low

            x_range = RangeInt(start=box[0], end=min(img_width, box[2]))
            y_range = RangeInt(start=box[1], end=min(img_height, box[3]))
            if x_range.span <= 0 or y_range.span <= 0:
                continue  # Invalid detection

            if isinstance(min_dims, BoxDimensions):
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
            if point is not None and len(point) >= 2:
                le_point = Point(x=point[0][0], y=point[0][1])
                re_point = Point(x=point[1][0], y=point[1][1])
                # TODO Decide whether to discard the face or to not include the eyes
                if le_point not in bounding_box or re_point not in bounding_box:
                    le_point = None
                    re_point = None

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=float(prob),
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFound("No face detected. Check the input image.")

        return DetectorBase.Outcome(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
