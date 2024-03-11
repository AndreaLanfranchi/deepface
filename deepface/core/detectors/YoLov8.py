from typing import Any, List

import os
import numpy
import gdown

from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import MissingOptionalDependency

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    what: str = f"{__name__} requires `ultralytics` library."
    what += "You can install by 'pip install ultralytics' "
    raise MissingOptionalDependency(what) from None

logger = Logger.get_instance()

# YoloV8 detector
class Detector(DetectorBase):

    _detector: Any
    # _LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):

        file_name = "yolov8n-face.pt"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

        if not os.path.isfile(weight_file):
            logger.info(f"Download : {file_name}")
            try:
                url: str = (
                    "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
                )
                gdown.download(url, weight_file, quiet=False, user_agent="Mozilla/5.0")
            except Exception as err:
                raise ValueError(
                    f"Exception while downloading Yolo weights from {self._WEIGHT_URL}."
                    f"You may consider to download it to {weight_file} manually."
                ) from err

        self._detector = YOLO(weight_file)

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:

        ret: List[FacialAreaRegion] = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return ret

        # Detect faces
        results = self._detector.predict(img, verbose=False, show=False, conf=0.25)[0]

        # For each face, extract the bounding box, the landmarks and confidence

        for result in results:

            if result.boxes is None or result.keypoints is None:
                continue

            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]
            left_eye = result.keypoints.xy[0][0].tolist()
            right_eye = result.keypoints.xy[0][1].tolist()

            # eyes are list of float, need to cast them tuple of int
            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            ret.append(facial_area)

        return ret
