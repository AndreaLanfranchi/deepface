from typing import Any, Union, List
import cv2
import numpy
from deepface.models.Detector import Detector, FacialAreaRegion


class FastMtCnnClient(Detector):
    """
    This class is used to detect faces using fast mtcnn face detector.
    Note! This is an optional detector, ensure the library is installed.

    See the following link for more information:
    https://github.com/timesler/facenet-pytorch
    Examples https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
    """

    _detector: Any

    def __init__(self):
        self.name = "FastMtCnn"
        self.__initialize()

    def __initialize(self):

        try:
            from facenet_pytorch import MTCNN as fast_mtcnn

            self._detector = fast_mtcnn(
                image_size=160,
                thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
                post_process=True,
                device="cpu",
                select_largest=False,  # return result in descending order
            )

        except ModuleNotFoundError as e:
            raise ImportError(
                "FastMtcnn is an optional detector, ensure the library is installed."
                "Please install using 'pip install facenet-pytorch' "
            ) from e

    def detect_faces(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with FastMtCnn

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # mtcnn expects RGB but OpenCV read BGR

        detections = self._detector.detect(
            img_rgb, landmarks=True
        )  # returns boundingbox, prob, landmark
        if detections is not None and len(detections) > 0:

            for current_detection in zip(*detections):
                x, y, w, h = self._xyxy_to_xywh(current_detection[0])
                confidence = current_detection[1]
                left_eye = current_detection[2][0]
                right_eye = current_detection[2][1]

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

    def _xyxy_to_xywh(self, xyxy: Union[list, tuple]) -> list:
        """
        Convert xyxy format to xywh format.
        """
        x, y = xyxy[0], xyxy[1]
        w = xyxy[2] - x + 1
        h = xyxy[3] - y + 1
        return [x, y, w, h]
