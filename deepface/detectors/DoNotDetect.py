from typing import List
import numpy
from deepface.models.Detector import Detector as DetectorBase, FacialAreaRegion


class Detector(DetectorBase):
    """
    This class is used to skip face detection. It is used when the user
    wants to use a pre-detected face.
    """

    def __init__(self):
        super().__init__()
        self._name = "DoNotDetect"

    def process(self, img: numpy.ndarray) -> List["FacialAreaRegion"]:
        return [
            FacialAreaRegion(
                0,
                0,
                img.shape[1],
                img.shape[0],
                None,
                None,
                None,
            )
        ]
