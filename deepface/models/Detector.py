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
        self._name: Optional[str] = None  # Must be filled by specialized classes

    @abstractmethod
    def process(
        self,
        img: numpy.ndarray,
        min_height: int = 0,
        min_width: int = 0,
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
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown

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

                module = importlib.import_module(
                    name=f"{Detectors.__name__}.{module_name}"
                )
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        if issubclass(obj, Detector) and obj is not Detector:
                            logger.debug(
                                f"Found {obj.__name__} class in module {module.__name__}"
                            )
                            key_value: str = str(module.__name__.split(".")[-1]).lower()
                            available_detectors[key_value] = obj
                            break  # Only one detector per module

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
            logger.critical(f"Failed to instantiate detector : {name} Error: {str(ex)}")
            raise ex

        return instance


class Point:
    """
    This class is used to represent a point in a 2D space
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __repr__(self):
        return self.__str__()


class RangeInt:
    """
    This class is used to represent a range of integers as [start, end]
    """

    def __init__(self, start: int, end: int):
        self.start = max(start, 0)
        self.end = max(end, 0)
        self.end = max(self.end, self.start)

    def __str__(self):
        return f"RangeInt(start={self.start}, end={self.end})"

    def __repr__(self):
        return self.__str__()


class DetectionResult:
    """
    This class is used to represent the result of a face detection.
    It contains the detected facial area and the confidence of the detection.
    """

    def __init__(
        self,
        detector: Detector,
        source: numpy.ndarray,
        y: RangeInt,
        x: RangeInt,
        left_eye: Optional[Point] = None,
        right_eye: Optional[Point] = None,
        confidence: Optional[float] = None,
    ):
        self._detector = detector
        self._source = source
        self._y = y
        self._x = x
        self._left_eye = left_eye
        self._right_eye = right_eye
        self._confidence = confidence

    @property
    def height(self) -> int:
        return self._y.end - self._y.start

    @property
    def width(self) -> int:
        return self._x.end - self._x.start

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def empty(self) -> bool:
        return self.area == 0

    @property
    def face(self) -> numpy.ndarray:
        return self._source[self._y.start : self._y.end, self._x.start : self._x.end]

    @property
    def confidence(self) -> Optional[float]:
        return self._confidence


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
