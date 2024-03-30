from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

import time
import numpy

from deepface.core.types import DetectedFace, BoxDimensions
from deepface.core.colors import KBGR_COLOR_BOUNDING_BOX
from deepface.core import reflection, detectors
from deepface.commons.logger import Logger

logger = Logger.get_instance()


# Abstract class all specialized face detectors must inherit from.
# A face detection consists in finding [0,inf) faces in an image and
# returning the region of the image where the face is located.
class Detector(ABC):
    """
    Interface for in picture face detection.
    Detects face(s) in an image and returns the bounding box(es) and landmarks.
    """

    _name: Optional[str] = None  # Must be filled by specialized classes

    @dataclass(frozen=True)
    class Results:
        detector: str
        img: numpy.ndarray
        tag: Optional[str]
        detections: list[DetectedFace]

        def __post_init__(self):
            assert isinstance(self.detector, str)
            assert isinstance(self.img, numpy.ndarray)
            assert self.tag is None or isinstance(self.tag, str)
            assert isinstance(self.detections, list)
            if self.detector.strip() == "":
                raise ValueError("Detector name must be non-empty")

        def __bool__(self):
            return len(self.detections) > 0

        def __len__(self):
            return len(self.detections)

        def plot(
            self,
            index: Optional[int] = None,
            copy: bool = False,
            color: Tuple[int, int, int] = KBGR_COLOR_BOUNDING_BOX,
            thickness: int = 2,
            key_points: bool = False,
        ) -> numpy.ndarray:
            """
            Draw the detected face(s) boundaries and landmarks on the image.

            Args:
                index (Optional[int]): Index of the face detection to plot.
                  Omitting the index will plot all detections (default: None)
                copy (bool): Whether to return the drawings on a copy of the image (default: False)
                color (Tuple[int, int, int]): BGR color code for drawing the bounding box
                  (default: KCOLOR_BGR_CYAN)
                thickness (int): Thickness of the bounding box (default: 2)
                eyes (bool): Whether to draw eye landmarks (default: False)

            Returns:
                numpy.ndarray: The image with the detected faces plotted.

            Raises:
                IndexError: If the index is out of bounds
            """
            if copy:
                img = self.img.copy()
            else:
                img = self.img

            if index is not None:
                if index < 0 or index >= len(self.detections):
                    raise IndexError("Invalid index")
                detection = self.detections[index]
                img = detection.plot(
                    img=img, copy=False, color=color, thickness=thickness, key_points=key_points
                )
            else:
                for detection in self.detections:
                    img = detection.plot(
                        img=img, copy=False, color=color, thickness=thickness, key_points=key_points
                    )
            return img

        def crop_faces(self, index: Optional[int] = None) -> List[numpy.ndarray]:
            """
            Crop the detected face(s) from the original image.

            Args:
                results (Results): The face detection results.

            Returns:
                List[numpy.ndarray]: A list of cropped face images.

            Raises:
                IndexError: If the provided index is out of bounds
                Any exceptions raised by the `crop` method of the `DetectedFace` class.
            """
            if index is not None:
                if index < 0 or index >= len(self.detections):
                    raise IndexError("Invalid index")
                return [self.detections[index].crop(self.img)]

            return [detection.crop(self.img) for detection in self.detections]

    @property
    def name(self) -> str:
        if (
            self._name is None
            or not isinstance(self._name, str)
            or len(self._name) == 0
        ):
            return "<undefined>"
        return self._name

    @abstractmethod
    def process(
        self,
        img: numpy.ndarray,
        tag: Optional[str] = None,
        min_dims: BoxDimensions = BoxDimensions(0, 0),
        min_confidence: float = float(0.0),
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> Results:
        """
        Detect faces in the given image.

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array
            tag (Optional[str]): optional tag to identify the image
            min_dims: Optional[BoxDimensions]: filter to discard
              boxes around faces that are smaller than the given dimensions
            min_confidence (float): minimum confidence level for the detection
              Default is 0.0.
            key_points (bool): whether to detect facial key points. Default is True.
            raise_notfound (bool): if True, raise an exception if no faces are found
              Default is False.

        Returns:
            An instance of `Results`

        Raises:
            `TypeError`: If the image is not a valid numpy array
            `ValueError`: If the image is empty
            'RuntimeError': If the detector does not find any face in the image
                and the `raise_notfound` argument is True
        """

        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array")

        if tag is not None and not isinstance(tag, str):
            raise TypeError("Tag must be a valid string")

        if len(img.shape) != 3:
            raise ValueError("Image must be a 3D numpy array")

        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Image must be non-empty")

        if not isinstance(min_dims, BoxDimensions):
            raise TypeError("Min dims must be a valid BoxDimensions object")

        if not isinstance(min_confidence, float):
            raise TypeError("Min confidence must be a valid float")

        if not isinstance(raise_notfound, bool):
            raise TypeError("Raise not found must be a valid boolean")

        if min_confidence < 0.0:
            raise ValueError("Min confidence must be non-negative")

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
        return "yunet"

    @staticmethod
    def instance(name: Optional[str] = None, singleton: bool = True) -> "Detector":
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
