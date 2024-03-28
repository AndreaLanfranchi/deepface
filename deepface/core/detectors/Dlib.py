from typing import Any, List, Optional

import os
import bz2
import shutil
import cv2
import gdown
import numpy

from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)

from deepface.core.exceptions import FaceNotFoundError, MissingDependencyError
from deepface.core.detector import Detector as DetectorBase
from deepface.commons import folder_utils
from deepface.commons.logger import Logger

try:
    import dlib
except ModuleNotFoundError:
    what: str = f"{__name__} requires `dlib` library."
    what += "You can install by 'pip install dlib' "
    raise MissingDependencyError(what) from None

logger = Logger.get_instance()


# Dlib detector (optional)
class Detector(DetectorBase):

    _detector: Any
    _predictor: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        file_name = "shape_predictor_5_face_landmarks.dat"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

        # check required file exists in the home/.deepface/weights folder
        if os.path.isfile(weight_file) != True:

            logger.info(f"Download : {file_name}")
            source_file = f"{file_name}.bz2"

            url = f"http://dlib.net/files/{source_file}"
            dest = os.path.join(folder_utils.get_weights_dir(), source_file)
            gdown.download(url, dest, quiet=False)

            with bz2.BZ2File(dest, "rb") as zipfile, open(weight_file, "wb") as f:
                shutil.copyfileobj(zipfile, f)

            os.remove(dest)

        # dlib's HOG + Linear SVM face detector
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(weight_file)

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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces: List[DetectedFace] = []

        # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
        rects, scores, _ = self._detector.run(gray, 1)
        assert len(rects) == len(scores)

        for rect, score in zip(rects, scores):
            if min_confidence is not None and score < min_confidence:
                continue

            x_range = RangeInt(rect.left(), min(rect.right(), img_width))
            y_range = RangeInt(rect.top(), min(rect.bottom(), img_height))
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

            if detect_eyes:
                shape = self._predictor(gray, rect)
                if shape.num_parts == 5:
                    # For dlib’s 5-point facial landmark detector
                    # Left eye: parts 0, 1
                    # Right eye: parts 2, 3
                    # Nose: part 4 (actually not used)

                    # For each eye, we will use the average of the two points
                    le_point = Point(
                        x=(shape.part(0).x + shape.part(1).x) // 2,
                        y=(shape.part(0).y + shape.part(1).y) // 2,
                    )
                    re_point = Point(
                        x=(shape.part(2).x + shape.part(3).x) // 2,
                        y=(shape.part(2).y + shape.part(3).y) // 2,
                    )
                    if le_point not in bounding_box or re_point not in bounding_box:
                        le_point = None
                        re_point = None

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=float(score),
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
