import os
from typing import Any, List
import numpy
import gdown
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons import folder_utils
from deepface.commons.logger import Logger

logger = Logger()

# Model's weights paths
PATH = "/.deepface/weights/yolov8n-face.pt"

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URL = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"

# Confidence thresholds for landmarks detection
# used in alignment_procedure function
LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

class YoloClient(Detector):
    def __init__(self):
        self.name = "yolov8"
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. \
                Please install using 'pip install ultralytics' "
            ) from e


        file_name = "yolov8n-face.pt"
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        # Download the model's weights if they don't exist
        if not os.path.isfile(output):
            logger.info(f"Download : {file_name}")
            try:
                gdown.download(WEIGHT_URL, output, quiet=False)
            except Exception as err:
                raise ValueError(
                    f"Exception while downloading Yolo weights from {WEIGHT_URL}."
                    f"You may consider to download it to {output} manually."
                ) from err

        # Return face_detector
        return YOLO(output)

    def detect_faces(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with yolo

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        # Detect faces
        results = self.model.predict(img, verbose=False, show=False, conf=0.25)[0]

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
            results.append(facial_area)

        return results
