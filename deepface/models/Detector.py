from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import importlib
import inspect
import pkgutil

import numpy

# Abstract class all specialized face detectors must inherit from.
# A face detection consists in finding [0,inf) faces in an image and
# returning the region of the image where the face is located.
class Detector(ABC):

    def __init__(self):
        self._name: Optional[str] = None # Must be filled by specialized classes

    @abstractmethod
    def process(self, img: numpy.ndarray) -> List["FacialAreaRegion"]:
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

    @property
    def name(self) -> str:
        return "<undefined>" if self._name is None else self._name

    @staticmethod
    def instance(name: str, singleton: bool = False) -> "Detector":
        """
        Returns a new instance of a detector matching the given name.

        Args:
            name (str): The name of the detector to instantiate
            singleton (bool): If True, the same instance will be returned
                for the same name. If False, a new instance will be returned
                each time.

        Returns:
            detector (Detector): A new instance of the detector

        Raises:
            ValueError: If the detector name empty
            KeyError: If the detector name is unknown

        """
        name = name.lower().strip()
        if len(name) == 0:
            raise ValueError("Empty detector name")

        return None

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
