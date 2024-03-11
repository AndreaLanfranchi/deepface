from typing import List

import numpy
from retinaface import RetinaFace as rf

from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion

# RetinaFace detector
class Detector(DetectorBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._detector = rf.build_model()

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
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
