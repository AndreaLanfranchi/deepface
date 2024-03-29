from typing import Any, List, Optional, Sequence

import os
import cv2
import numpy

from cv2.typing import MatLike, Rect
from deepface.core.detector import Detector as DetectorBase
from deepface.core.exceptions import FaceNotFoundError
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)


# OpenCV's detector (default)
class Detector(DetectorBase):

    _detector: cv2.CascadeClassifier
    _eye_detector: cv2.CascadeClassifier

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        self._detector = self._build_cascade("haarcascade")
        self._eye_detector = self._build_cascade("haarcascade_eye")

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
        img_height, img_width = img.shape[:2]

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces: List[DetectedFace] = []

        # note that, by design, opencv's haarcascade scores are >0 but not capped at 1
        # TODO : document values and magic numbers
        faces, _, weights = self._detector.detectMultiScale3(
            image=gray_img,
            scaleFactor=1.1,
            minNeighbors=10,
            outputRejectLevels=True,
        )

        for rect, weight in zip(faces, weights):
            if min_confidence is not None and float(weight) < min_confidence:
                continue

            x, y, w, h = rect
            x_range = RangeInt(int(x), int(min(x + w, img_width)))
            y_range = RangeInt(int(y), int(min(y + h, img_height)))
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

            cropped_img = gray_img[
                bounding_box.top_left.y : bounding_box.bottom_right.y,
                bounding_box.top_left.x : bounding_box.bottom_right.x,
            ]

            le_point = None
            re_point = None
            if detect_eyes:
                eyes: List[Point] = self.find_eyes(cropped_img)
                if len(eyes) == 2:
                    # Normalize left and right eye coordinates to the whole image
                    # We swap the eyes because the first eye is the right one
                    re_point = Point(
                        x=eyes[0].x + bounding_box.top_left.x,
                        y=eyes[0].y + bounding_box.top_left.y,
                    )
                    le_point = Point(
                        x=eyes[1].x + bounding_box.top_left.x,
                        y=eyes[1].y + bounding_box.top_left.y,
                    )
                    if le_point not in bounding_box or re_point not in bounding_box:
                        le_point = None
                        re_point = None

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=float(weight),
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=str(self._name),
            img=img,
            detections=detected_faces,
        )

    def find_eyes(self, img: MatLike) -> List[Point]:

        ret: List[Point] = []
        rects: Sequence[Rect] = self._eye_detector.detectMultiScale(
            image=img,
            minNeighbors=int(10),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(rects) < int(2):
            return ret

        # ----------------------------------------------------------------

        # opencv eye detection module is not strong. it might find more than 2 eyes!
        # besides, it returns eyes with different order in each call (issue 435)
        # this is an important issue because opencv is the default detector and ssd also uses this
        # find the largest 2 eye. Thanks to @thelostpeace

        rects: Sequence[Rect] = sorted(
            rects, key=lambda v: abs(v[2] * v[3]), reverse=True
        )[:2]

        # Eventually, we have 2 eyes which we order left to right by x coordinate
        rects: Sequence[Rect] = sorted(rects, key=lambda v: v[0])

        x, y, w, h = (int(val) for val in rects[0])
        left_box = BoundingBox(
            top_left=Point(x=x, y=y),
            bottom_right=Point(x=x + w, y=y + h),
        )

        x, y, w, h = (int(val) for val in rects[1])
        right_box = BoundingBox(
            top_left=Point(x=x, y=y),
            bottom_right=Point(x=x + w, y=y + h),
        )

        left_eye = left_box.center
        right_eye = right_box.center

        return [left_eye, right_eye]

    def _build_cascade(self, model_name="haarcascade") -> Any:

        match model_name:
            case "haarcascade":
                file_name = "haarcascade_frontalface_default.xml"
            case "haarcascade_eye":
                file_name = "haarcascade_eye.xml"
            case _:
                raise NotImplementedError(f"Unknown : {model_name}")

        cv2_root = os.path.dirname(cv2.__file__)
        file_path = os.path.join(cv2_root, "data", file_name)
        if os.path.isfile(file_path) != True:
            raise RuntimeError(
                f"Coulnd't find {file_path}\n" "Check opencv is installed properly"
            )

        return cv2.CascadeClassifier(file_path)
