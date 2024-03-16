from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import time
import numpy

from deepface.core.types import InPictureFace, BoxDimensions
from deepface.core import reflection, detectors
from deepface.commons.logger import Logger

logger = Logger.get_instance()


# Abstract class all specialized face detectors must inherit from.
# A face detection consists in finding [0,inf) faces in an image and
# returning the region of the image where the face is located.
class Detector(ABC):
    """
    Interface in picture face detection.

    Args:
        img (numpy.ndarray): pre-loaded image as numpy array
        min_dims: Optional[BoxDimensions]: filter to discard
        boxes around faces that are smaller than the given dimensions
        min_confidence (float): minimum confidence level for the detection

    Returns:
        `List[InPictureFace]`: A list of the regions containing a face in
        the image provided. The list may be empty if no faces are detected.

    Raises:
        `ValueError`: If the image is not a valid numpy array
    """

    _name: Optional[str] = None  # Must be filled by specialized classes

    @abstractmethod
    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.0,
    ) -> List[InPictureFace]:

        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array")

        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Image must be non-empty")

        if min_dims is not None:
            if not isinstance(min_dims, BoxDimensions):
                raise TypeError("Min dims must be a valid BoxDimensions object")
            if min_dims.width == 0 and min_dims.height == 0:
                raise ValueError(
                    "At least one dimension from min_dims must be non-zero"
                )

        min_confidence = abs(float(min_confidence))

    @property
    def name(self) -> str:
        if (
            self._name is None
            or not isinstance(self._name, str)
            or len(self._name) == 0
        ):
            return "<undefined>"
        return self._name

    @staticmethod
    def get_available_detectors() -> List[str]:
        """
        Get the names of the available face detectors.

        Returns:
            A list of strings representing the names of the available face detectors.
        """
        global available_detectors
        if not "available_detectors" in globals():
            available_detectors = reflection.get_derived_classes(
                package=detectors, base_class=Detector
            )
        return list(available_detectors.keys())

    @staticmethod
    def get_default() -> str:
        """
        Get the default face detector name.

        Returns:
            The name of the default face detector.
        """
        return "opencv"

    @staticmethod
    def instance(name: Optional[str], singleton: bool = True) -> "Detector":
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
        if name is None:
            name = Detector.get_default()
        elif not isinstance(name, str):
            raise TypeError(
                f"Invalid 'name' argument type [{type(name).__name__}] : expected str"
            )
        if not isinstance(singleton, bool):
            raise TypeError(
                f"Invalid 'singleton' argument type [{type(singleton).__name__}] : expected bool"
            )

        name = name.lower().strip()
        if len(name) == 0:
            name = Detector.get_default()

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
