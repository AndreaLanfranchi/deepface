from typing import Any, Dict, List, Optional

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
        self._KDEFAULT_MIN_CONFIDENCE = float(0.4)
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
        tag: Optional[str] = None,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: Optional[float] = None,
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)

        if min_dims is None:
            min_dims = BoxDimensions(width=0, height=0)
        if min_confidence is None:
            min_confidence = self._KDEFAULT_MIN_CONFIDENCE
            raise ValueError(
                f"min_confidence must be in the range [0, 1]. Got {min_confidence}."
            )

        img_height, img_width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces: List[DetectedFace] = []

        # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
        rects, scores, _ = self._detector.run(gray, 1)
        assert len(rects) == len(scores)

        for rect, score in zip(rects, scores):

            confidence = float(score)
            if confidence < min_confidence:
                continue

            x_range = RangeInt(rect.left(), min(rect.right(), img_width))
            y_range = RangeInt(rect.top(), min(rect.bottom(), img_height))
            if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                continue  # Invalid or empty detection

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            points: Optional[Dict[str, Optional[Point]]] = None
            if key_points:
                points = dict[str, Optional[Point]]()
                # For dlib’s 5-point facial landmark detector
                # Left eye: parts 0, 1
                # Right eye: parts 2, 3
                # Nose: part 4
                shape = self._predictor(gray, rect)
                if shape.num_parts >= 4:
                    le_point = Point(
                        x=(shape.part(0).x + shape.part(1).x) // 2,
                        y=(shape.part(0).y + shape.part(1).y) // 2,
                    )
                    re_point = Point(
                        x=(shape.part(2).x + shape.part(3).x) // 2,
                        y=(shape.part(2).y + shape.part(3).y) // 2,
                    )
                    points.update({"lec": le_point, "rec": re_point})
                if shape.num_parts == 5:
                    n_point = Point(
                        x=shape.part(4).x,
                        y=shape.part(4).y,
                    )
                    points.update({"nt": n_point})

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
