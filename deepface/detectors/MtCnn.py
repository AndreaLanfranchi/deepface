from typing import List
import numpy
from mtcnn import MTCNN
from deepface.models.Detector import Detector, FacialAreaRegion


class MtCnnClient(Detector):
    """
    This class is used to detect faces using MtCnn face detector.
    """

    def __init__(self):
        super().__init__()
        self._name = "MtCnn"
        self._detector = MTCNN()

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with mtcnn

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """

        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        # mtcnn expects RGB but OpenCV read BGR
        # TODO: Verify if the image is in the right BGR format
        # before converting it to RGB
        img_rgb = img[:, :, ::-1]

        detections = self._detector.detect_faces(img_rgb)

        if detections is not None and len(detections) > 0:

            for current_detection in detections:
                x, y, w, h = current_detection["box"]
                confidence = current_detection["confidence"]
                left_eye = current_detection["keypoints"]["left_eye"]
                right_eye = current_detection["keypoints"]["right_eye"]

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
