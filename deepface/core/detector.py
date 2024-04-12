from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import time
import numpy

from deepface.core import imgutils
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

    _name: Optional[str] = None  # Must be filled by derived classes
    _KDEFAULT_MIN_CONFIDENCE: float = 0.0  # Every detector must define its own

    @dataclass(frozen=True)
    class Results:
        detector: str
        img: numpy.ndarray
        tag: Optional[str]
        detections: list[DetectedFace]

        def __post_init__(self):

            if not isinstance(self.detector, str):
                raise TypeError("Detector name must be a valid string")
            if self.detector not in Detector.get_available_detectors():
                raise ValueError(f"Invalid or unknown detector {self.detector}")
            if not imgutils.is_valid_image(self.img):
                raise ValueError("Invalid or empty image")
            if self.tag is not None:
                if not isinstance(self.tag, str):
                    raise TypeError("Tag must be a valid string")
            if not isinstance(self.detections, list):
                what: str = "Invalid detections type. Expected list "
                what += f"got {type(self.detections).__name__}"
                raise TypeError(what)
            for item in self.detections:
                if not isinstance(item, DetectedFace):
                    what: str = "Invalid detection type. Expected DetectedFace "
                    what += f"got {type(item).__name__}"
                    raise TypeError(what)

        def __bool__(self):
            return len(self.detections) > 0

        def __len__(self):
            return len(self.detections)

        def plot(
            self,
            items: Optional[Union[int, List[int]]] = None,
            copy: bool = False,
            color: Tuple[int, int, int] = KBGR_COLOR_BOUNDING_BOX,
            thickness: int = 2,
            key_points: bool = False,
        ) -> numpy.ndarray:
            """
            Draw the detected face(s) boundaries and landmarks on the image.

            Args:
                `items` : Index(es) of the face detection to plot.
                  Omitting the index will plot all detections (default: None)
                copy (bool): Whether to return the drawings on a copy of the image (default: False)
                color (Tuple[int, int, int]): BGR color code for drawing the bounding box
                  (default: KCOLOR_BGR_CYAN)
                thickness (int): Thickness of the bounding box (default: 2)
                eyes (bool): Whether to draw eye landmarks (default: False)

            Returns:
                numpy.ndarray: The image with the detected faces plotted.

            Raises:
                TypeError: If the index is not an integer
                IndexError: If the index is out of bounds
            """

            if items is None:
                items = list(range(len(self.detections)))
            elif isinstance(items, int):
                items = [
                    items,
                ]
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, int):
                        what: str = "Invalid index type. Expected int "
                        what += f"got {type(item).__name__}"
                        raise TypeError(what)
                    if item < 0 or item >= len(self.detections):
                        raise IndexError("Out of bounds index")
            else:
                what: str = (
                    "Invalid [items] argument type. Expected [int | list of int | None] "
                )
                what += f"got {type(items).__name__}"
                raise TypeError(what)

            if copy:
                img = self.img.copy()
            else:
                img = self.img

            for item in items:
                detection = self.detections[item]
                img = detection.plot(
                    img=img,
                    copy=False,
                    color=color,
                    thickness=thickness,
                    key_points=key_points,
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
        return self._name.lower().strip()

    @property
    def default_min_confidence(self) -> float:
        return self._KDEFAULT_MIN_CONFIDENCE

    @abstractmethod
    def process(
        self,
        img: numpy.ndarray,
        tag: Optional[str] = None,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: Optional[float] = None,
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

        if not imgutils.is_valid_image(img):
            raise ValueError("Invalid image or empty image")

        if tag is not None and not isinstance(tag, str):
            raise TypeError("Tag must be a valid string")

        if min_dims is not None and not isinstance(min_dims, BoxDimensions):
            raise TypeError(
                "Min dims must be a valid BoxDimensions class object or None"
            )

        if min_confidence is not None and not isinstance(min_confidence, float):
            raise TypeError("Min confidence must be a valid float or None")

        if not isinstance(raise_notfound, bool):
            raise TypeError("Raise not found must be a valid boolean")

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
    def default() -> str:
        """
        Get the default face detector name.

        Returns:
        --------
            A string
        """
        return "fastmtcnn"

    @staticmethod
    def instance(
        name_or_inst: Union[str, "Detector"] = "default",
        singleton: bool = True,
    ) -> "Detector":
        """
        `Detector` "lazy" factory method.

        Args:
        -----
            `name_or_inst`: A string representing the name of the detector to instantiate
              or an instance of a `Detector` subclass. When a `Detector` instance is given,
              the same instance is returned. When a string is given this cases are handled:
              - If the string equals `default`, the default detector is assumed
              (see `Detector.default()` method) and returned
              - If the string is a known detector name, an instance of the corresponding
              detector is returned

            `singleton (bool)`: If True, the factory will return a singleton instance of the
              detector. If False, a new instance is returned every time the factory is called

        Returns:
        --------
            An instance of the `Detector` subclass matching the given name

        Raises:
        -------
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown
            `ImportError`: If the detector instance cannot be instantiated
        """

        if isinstance(name_or_inst, Detector):
            return name_or_inst

        if not isinstance(name_or_inst, str):
            what: str = "Invalid 'name_or_inst' argument type. Expected str "
            what += f"got {type(name_or_inst).__name__}"
            raise TypeError(what)

        name_or_inst = name_or_inst.lower().strip()

        if len(name_or_inst) == 0:
            what: str = "Invalid 'name_or_inst' argument value."
            what += " Expected a valid detector name or `default`. Got empty string"
            raise ValueError(what)

        if name_or_inst == "default":
            name_or_inst = Detector.default()

        if not isinstance(singleton, bool):
            raise TypeError(
                f"Invalid 'singleton' argument type [{type(singleton).__name__}] : expected bool"
            )

        name_or_inst = name_or_inst.lower().strip()
        if len(name_or_inst) == 0:
            name_or_inst = Detector.default()

        global detectors_instances  # singleton design pattern
        if not "detectors_instances" in globals():
            detectors_instances = {}

        global available_detectors
        if not "available_detectors" in globals():
            available_detectors = reflection.get_derived_classes(
                package=detectors, base_class=Detector
            )

        if name_or_inst not in available_detectors.keys():
            raise NotImplementedError(f"Unknown detector [{name_or_inst}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_detectors[name_or_inst]()
                logger.debug(
                    f"Instantiated detector [{name_or_inst}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name_or_inst not in detectors_instances.keys():
                    detectors_instances[name_or_inst] = available_detectors[
                        name_or_inst
                    ]()
                    logger.debug(
                        f"Instantiated detector [{name_or_inst}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = detectors_instances[name_or_inst]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on detector [{name_or_inst}] Error: {ex.args}"
            )
            raise ex

        return instance
