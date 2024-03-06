import time
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import importlib
import inspect
import pkgutil

import numpy
from deepface import detectors as Detectors
from deepface.commons.logger import Logger
logger = Logger()


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
    def instance(name: str, singleton: bool = True) -> "Detector":
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
            NotImplementedError: If the detector name is unknown

        """
        name = name.lower().strip()
        if len(name) == 0:
            raise ValueError("Empty detector name")

        global detectors_instances  # singleton design pattern
        if not "detectors_instances" in globals():
            detectors_instances = {}

        global available_detectors
        if not "available_detectors" in globals():
            available_detectors = {}
            for _, module_name, _ in pkgutil.walk_packages(Detectors.__path__):

                # TODO : Remove this when DetectorWrapper is removed
                if __name__.endswith(module_name):
                    continue  # Don't import self

                module = importlib.import_module(name=f"{Detectors.__name__}.{module_name}")
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        if issubclass(obj, Detector) and obj is not Detector:
                            logger.debug(
                                f"Found {obj.__name__} class in module {module.__name__}"
                            )
                            key_value: str = str(module.__name__.split(".")[-1]).lower()
                            available_detectors[key_value] = obj
                            break # Only one detector per module

        if name not in available_detectors.keys():
            raise NotImplementedError(f"Unknown detector : {name}")

        tic = time.time()
        try:
            if not singleton:
                instance = available_detectors[name]()
                logger.debug(
                    f"Instantiated detector : {name} ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name not in detectors_instances.keys():
                    detectors_instances[name] = available_detectors[name]()
                    logger.debug(
                        f"Instantiated detector : {name} ({time.time() - tic:.3f} seconds)"
                    )
                instance = detectors_instances[name]
        except Exception as ex:
            logger.critical(
                f"Failed to instantiate detector : {name} Error: {str(ex)}"
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
