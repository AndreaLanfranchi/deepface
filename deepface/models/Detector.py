from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy

# Notice that all facial detector models must inherit from this class
# and implement the abstract methods.
class Detector(ABC):
    
    name: str               # Name of the detector

    @abstractmethod
    def detect_faces(self, img: numpy.ndarray) -> List["FacialAreaRegion"]:
        """
        Interface in picture face detection.

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
                where each object contains:

            - facial_area (FacialAreaRegion): The facial area region represented
                as x, y, w, h, left_eye and right_eye
        """
        pass

class DonotDetect(Detector):
    """
    This class is used to skip face detection. It is used when the user
    wants to use a pre-detected face.
    """
    def __init__(self):
        self.name = "DonotDetect"

    def detect_faces(self, img: numpy.ndarray) -> List["FacialAreaRegion"]:
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

class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]]
    right_eye: Optional[Tuple[int, int]]
    confidence: Optional[float]

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.confidence = confidence


class DetectedFace:
    img: numpy.ndarray
    facial_area: FacialAreaRegion

    def __init__(self, img: numpy.ndarray, facial_area: FacialAreaRegion):
        self.img = img
        self.facial_area = facial_area
