from typing import Any, Dict, List, Optional, Sequence

import os
import cv2
import numpy

from cv2.typing import MatLike, Rect
from deepface.core.imgutils import is_grayscale_image
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

# OpenCV's detector (default)
class Detector(DetectorBase):

    _detector: cv2.CascadeClassifier
    _eye_detector: cv2.CascadeClassifier

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._KDEFAULT_MIN_CONFIDENCE = float(0.8)
        self._initialize()

    def _initialize(self):
        self._detector = self._build_cascade("haarcascade")
        self._eye_detector = self._build_cascade("haarcascade_eye")

    def process(
        self,
        img: numpy.ndarray,
        tag: Optional[str] = None,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: Optional[float] = None, # See notes below
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)

        if min_dims is None:
            min_dims = BoxDimensions(width=0, height=0)
        if min_confidence is None:
            min_confidence = self._KDEFAULT_MIN_CONFIDENCE
        if min_confidence < 0 or min_confidence > 1:
            raise ValueError(
                f"min_confidence must be in the range [0, 1]. Got {min_confidence}."
            )

        detected_faces: List[DetectedFace] = []
        img_height, img_width = img.shape[:2]

        # note that, by design, opencv's haarcascade scores are >0 but not capped at 1
        # TODO : document values and magic numbers
        faces, _, weights = self._detector.detectMultiScale3(
            image=img,
            scaleFactor=1.1,
            minNeighbors=10,
            outputRejectLevels=True,
        )

        for rect, weight in zip(faces, weights):

            # We normalize weight to [0, 1] range
            # considering an optimum value of 5.0
            confidence = round(float(min(weight / 5.0, 1.0)), 5)
            if confidence < min_confidence:
                continue

            x, y, w, h = rect
            x_range = RangeInt(int(x), int(min(x + w, img_width)))
            y_range = RangeInt(int(y), int(min(y + h, img_height)))
            if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                continue  # Invalid or empty detection

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            points: Optional[Dict[str, Point]] = None
            if key_points:
                cropped_img = img[
                    bounding_box.top_left.y : bounding_box.bottom_right.y,
                    bounding_box.top_left.x : bounding_box.bottom_right.x,
                ]
                eyes: List[Point] = self.find_eyes(cropped_img)
                if len(eyes) == 2:
                    # Normalize left and right eye coordinates to the whole image
                    re_point = Point(
                        x=eyes[0].x + bounding_box.top_left.x,
                        y=eyes[0].y + bounding_box.top_left.y,
                    )
                    le_point = Point(
                        x=eyes[1].x + bounding_box.top_left.x,
                        y=eyes[1].y + bounding_box.top_left.y,
                    )
                    points = {"lec": le_point, "rec": re_point}

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
            detector=str(self.name),
            img=img,
            tag=tag,
            detections=detected_faces,
        )

    def find_eyes(self, img: MatLike) -> List[Point]:

        ret: List[Point] = []
        if not is_grayscale_image(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects: Sequence[Rect] = self._eye_detector.detectMultiScale(
            image=img,
            scaleFactor=float(1.1),
            minNeighbors=int(10),
        )

        # We don't actually care about their order.
        # They will be eventually ordered during the
        # addition to the detected face.
        for i in range(min(2, len(rects))):
            x, y, w, h = (int(round(val)) for val in rects[i])
            ret.append(Point(x=x + w // 2, y=y + h // 2))

        return ret

    def _build_cascade(self, model_name="haarcascade") -> Any:

        match model_name:
            case "haarcascade":
                file_name = "haarcascade_frontalface_default"
            case "haarcascade_eye":
                # file_name = "haarcascade_eye"
                file_name = "haarcascade_eye_tree_eyeglasses"
            case _:
                raise NotImplementedError(f"Unknown : {model_name}")

        cv2_root = os.path.dirname(cv2.__file__)
        file_path = os.path.join(cv2_root, "data", f"{file_name}.xml")
        if os.path.isfile(file_path) != True:
            raise RuntimeError(
                f"Coulnd't find {file_path}\n" "Check opencv is installed properly"
            )

        return cv2.CascadeClassifier(file_path)
