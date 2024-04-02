from typing import Dict, List, Optional

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
from deepface.core.exceptions import FaceNotFoundError, MissingDependencyError
from deepface.commons.logger import Logger

try:
    from facenet_pytorch import MTCNN as fast_mtcnn
except ModuleNotFoundError:
    what: str = f"{__name__} requires `facenet-pytorch` library."
    what += "You can install by 'pip install facenet-pytorch' "
    raise MissingDependencyError(what) from None

logger = Logger.get_instance()


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
        tag: Optional[str] = None,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = float(0.99),
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)

        if min_dims is None:
            min_dims = BoxDimensions(width=0, height=0)

        img_height, img_width = img.shape[:2]
        detected_faces: List[DetectedFace] = []

        # TODO: check image has less than 4 channels

        # mtcnn expects RGB but OpenCV read BGR
        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, probs, keypoints_list = self._detector.detect(img_rgb, landmarks=True)

        for box, prob, keypoints in zip(boxes, probs, keypoints_list):

            confidence = float(prob)
            if confidence < min_confidence:
                continue  # Confidence too low

            if box is None or not isinstance(box, numpy.ndarray) or box.shape[0] != 4:
                continue  # No detection or tampered data

            x1, y1, x2, y2 = (int(round(val)) for val in box[:4])
            x_range = RangeInt(start=x1, end=min(x2, img_width))
            y_range = RangeInt(start=y1, end=min(y2, img_height))
            if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                continue  # Invalid or empty detection

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            points: Optional[Dict[str, Optional[Point]]] = None
            if (
                key_points
                and isinstance(keypoints, numpy.ndarray)
                and keypoints.shape[0] > 0
                and keypoints.shape[1] == 2  # 2D coordinates
            ):
                # 0: left eye
                # 1: right eye
                # 2: nose,
                # 3: left mouth
                # 4: right mouth

                points = dict[str, Optional[Point]]()
                if keypoints.shape[0] >= 2:
                    le_point = Point(
                        x=int(round(keypoints[1][0])),
                        y=int(round(keypoints[1][1])),
                    )
                    re_point = Point(
                        x=int(round(keypoints[0][0])),
                        y=int(round(keypoints[0][1])),
                    )
                    points.update({"lec": le_point, "rec": re_point})

                if keypoints.shape[0] >= 3:
                    n_point = Point(
                        x=int(round(keypoints[2][0])),
                        y=int(round(keypoints[2][1])),
                    )
                    points.update({"nt": n_point})

                if keypoints.shape[0] >= 5:
                    lm_point = Point(
                        x=int(round(keypoints[3][0])),
                        y=int(round(keypoints[3][1])),
                    )
                    rm_point = Point(
                        x=int(round(keypoints[4][0])),
                        y=int(round(keypoints[4][1])),
                    )
                    cm_point = Point(
                        x=(lm_point.x + rm_point.x) // 2,
                        y=(lm_point.y + rm_point.y) // 2,
                    )
                    points.update({"mlc": lm_point, "mrc": rm_point, "mc": cm_point})

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
