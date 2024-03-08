import time
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy
from deepface.core.types import uint32_t
from deepface.core import reflection, detectors
from deepface.commons.logger import Logger

logger = Logger.get_instance()


# Abstract class all specialized face detectors must inherit from.
# A face detection consists in finding [0,inf) faces in an image and
# returning the region of the image where the face is located.
class Detector(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes

    @abstractmethod
    def process(
        self,
        img: numpy.ndarray,
        min_height: uint32_t = 0,
        min_width: uint32_t = 0,
        min_confidence: float = 0.0,
    ) -> List["FacialAreaRegion"]:
        """
        Interface in picture face detection.

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array
            min_height (int): minimum height of the detected face (if any)
            min_width (int): minimum width of the detected face (if any)
            min_confidence (float): minimum confidence of the detected face (if any)

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
    def instance(name: str, singleton: bool = True) -> "Detector":
        """
        `Detector` factory method.

        Args:
            `name (str)`: The name of the detector to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Detector` subclass matching the given name

        Raises:
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown
            'ImportError': If the detector instance cannot be instantiated
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Invalid 'name' argument type [{type(name).__name__}] : expected str"
            )
        if not isinstance(singleton, bool):
            raise TypeError(
                f"Invalid 'singleton' argument type [{type(singleton).__name__}] : expected bool"
            )

        name = name.lower().strip()
        if len(name) == 0:
            raise KeyError("Empty detector name")

        global detectors_instances  # singleton design pattern
        if not "detectors_instances" in globals():
            detectors_instances = {}

        global available_detectors
        if not "available_detectors" in globals():
            available_detectors = reflection.get_derived_classes(
                package=detectors, base_class=Detector
            )

        if name not in available_detectors.keys():
            raise NotImplementedError(f"Unknown detector [{name}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_detectors[name]()
                logger.debug(
                    f"Instantiated detector [{name}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name not in detectors_instances.keys():
                    detectors_instances[name] = available_detectors[name]()
                    logger.debug(
                        f"Instantiated detector [{name}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = detectors_instances[name]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on detector [{name}] Error: {ex.args}"
            )
            raise ex

        return instance


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
