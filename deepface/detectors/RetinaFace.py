from typing import Any, List
import numpy
from retinaface import RetinaFace as rf
from deepface.models.Detector import Detector, FacialAreaRegion


class RetinaFaceClient(Detector):
    """
    This class is used to detect faces using RetinaFace.
    """

    _detector: Any

    def __init__(self):
        self.name = "RetinaFace"
        self._detector = rf.build_model()

    def detect_faces(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with retinaface

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        obj = rf.detect_faces(img, model=self._detector, threshold=0.9)

        if not isinstance(obj, dict):
            return results

        for face_idx in obj.keys():
            identity = obj[face_idx]
            detection = identity["facial_area"]

            y = detection[1]
            h = detection[3] - y
            x = detection[0]
            w = detection[2] - x

            # notice that these must be inverse for retinaface
            left_eye = identity["landmarks"]["right_eye"]
            right_eye = identity["landmarks"]["left_eye"]

            # eyes are list of float, need to cast them tuple of int
            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)

            confidence = identity["score"]

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
