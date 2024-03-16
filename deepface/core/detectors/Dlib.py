from typing import Any, List, Optional

import os
import bz2
import shutil
import gdown
import numpy

from deepface.core.types import BoxDimensions, InPictureFace, Point, RangeInt
from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase
from deepface.commons.logger import Logger

from deepface.core.exceptions import MissingOptionalDependency

try:
    import dlib
    from dlib import Detector as DlibDetector
except ModuleNotFoundError:
    what: str = f"{__name__} requires `dlib` library."
    what += "You can install by 'pip install dlib' "
    raise MissingOptionalDependency(what) from None

logger = Logger.get_instance()


# Dlib detector (optional)
class Detector(DetectorBase):

    _detector: DlibDetector
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
    ) -> List[InPictureFace]:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        results: List[InPictureFace] = []

        # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
        rects, scores, _ = self._detector.run(img, 1)
        assert len(rects) == len(scores)

        for rect, score in zip(rects, scores):
            if min_confidence is not None and score < min_confidence:
                continue

            x_range = RangeInt(rect.left(), min(rect.right(), img.shape[1]))
            y_range = RangeInt(rect.top(), min(rect.bottom(), img.shape[0]))

            if min_dims is not None:
                if min_dims.width > 0 and x_range.span < min_dims.width:
                    continue
                if min_dims.height > 0 and y_range.span < min_dims.height:
                    continue

            le_point = None
            re_point = None
            landmarks = self._predictor(img, rect)
            if len(landmarks) >= 3:
                le_point = Point(landmarks.part(2).x, landmarks.part(2).y)
                re_point = Point(landmarks.part(0).x, landmarks.part(0).y)

                # Martian positions ?
                # TODO Decide whether to discard the face or to not include the eyes
                if not x_range.contains(le_point.x) or not y_range.contains(le_point.y):
                    le_point = None
                if not x_range.contains(re_point.x) or not y_range.contains(re_point.y):
                    re_point = None

            results.append(
                InPictureFace(
                    detector=self.name,
                    source=img,
                    y_range=y_range,
                    x_range=x_range,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=score,
                )
            )

        return results
