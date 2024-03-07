import os
from typing import Any, List
import numpy
import gdown
from core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons import folder_utils
from deepface.commons.logger import Logger

logger = Logger()

# # Model's weights paths
# PATH = "/.deepface/weights/yolov8n-face.pt"

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
# WEIGHT_URL = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"

# Confidence thresholds for landmarks detection
# used in alignment_procedure function
# LANDMARKS_CONFIDENCE_THRESHOLD = 0.5


class Detector(DetectorBase):
    """
    This class is used to detect faces using YOLOv8 face detector.
    Note! This is an optional detector, ensure the library is installed.
    """

    _detector: Any
    _WEIGHT_URL: str = (
        "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
    )
    # _LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1))
        self.__initialize()

    def __initialize(self):

        try:
            from ultralytics import YOLO

            file_name = "yolov8n-face.pt"
            output = os.path.join(folder_utils.get_weights_dir(), file_name)

            # Download the model's weights if they don't exist
            if not os.path.isfile(output):
                logger.info(f"Download : {file_name}")
                try:
                    gdown.download(self._WEIGHT_URL, output, quiet=False)
                except Exception as err:
                    raise ValueError(
                        f"Exception while downloading Yolo weights from {self._WEIGHT_URL}."
                        f"You may consider to download it to {output} manually."
                    ) from err

        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. \
                Please install using 'pip install ultralytics' "
            ) from e

        self._detector = YOLO(output)

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with Yolov8

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        ret: List[FacialAreaRegion] = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return ret

        # Detect faces
        results = self._detector.predict(img, verbose=False, show=False, conf=0.25)[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:
            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            # left_eye_conf = result.keypoints.conf[0][0]
            # right_eye_conf = result.keypoints.conf[0][1]
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
