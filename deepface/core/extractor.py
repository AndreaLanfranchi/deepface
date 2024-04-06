from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import time
import cv2
import numpy

from deepface.core import extractors
from deepface.core import reflection
from deepface.core import imgutils
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace, Point
from deepface.commons.logger import Logger

logger = Logger.get_instance()


# Abstract class all specialized face extractors must inherit from.
# Creates the synthetic digital representation of a detected face.
# It is assumed the input picture is already a face previously detected.
class Extractor(ABC):
    """
    Interface for digital face(s) representation.
    """

    _name: Optional[str] = None  # Must be filled by derived classes
    _input_shape: BoxDimensions
    _output_shape: int

    @abstractmethod
    def process(
        self,
        img: numpy.ndarray,
        tag: Optional[str] = None,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> List[float]:
        """
        Extracts the digital representation of a face from an image.

        Args:
        ----
            `img` (numpy.ndarray): The image containing the face to be extracted
            `face` (Optional[BoundingBox, DetectedFace]): The face to be extracted.
                If None, the whole input image is assumed to be a face

        Returns:
        -------
            List[float]: The digital representation of the face

        Raises:
        ------
            ValueError: If the image or the bounding box is invalid
            TypeError: If the face argument is not of the expected type

        """

        if not imgutils.is_valid_image(img):
            raise ValueError("Invalid image")

        if tag is not None and not isinstance(tag, str):
            raise TypeError("Tag must be a valid string")

        if not face is None:
            if isinstance(face, BoundingBox):
                if 0 == face.area:
                    raise ValueError("Empty face bounding box")
            elif isinstance(face, DetectedFace):
                if 0 == face.bounding_box.area:
                    raise ValueError("Empty face bounding box")
            else:
                what: str = "Invalid face argument type"
                what += (
                    " expected [BoundingBox | DetectedFace | None], got "
                    + type(face).__name__
                )
                raise TypeError(what)

    def _to_required_shape(
        self,
        img: numpy.ndarray,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> numpy.ndarray:
        """
        Scales and pads the image to fit the input shape of the model.
        If `face` argument is provided the image is cropped to the face bounding box.
        Proportions are kept.

        Args:
        ----
            `img` (numpy.ndarray): The image to be scaled and padded
            `face` (Optional[BoundingBox, DetectedFace]): The face to be extracted.
                If None, the whole input image is assumed to be a face

        Returns:
        -------
            numpy.ndarray: The scaled and padded image

        Raises:
        -------
            ValueError: If the bounding box (when provided) exceeds
                the image dimensions
        """

        ret = numpy.zeros(
            shape=(self._input_shape.height, self._input_shape.width, 3),
            dtype=numpy.uint8,
        )

        bbox: Optional[BoundingBox] = None
        img_height: int = img.shape[0]
        img_width: int = img.shape[1]

        if face is None:
            # Assume the whole image is a face
            bbox = BoundingBox(
                top_left=Point(0, 0), bottom_right=Point(img_width, img_height)
            )
        elif isinstance(face, BoundingBox):
            bbox = face
        elif isinstance(face, DetectedFace):
            bbox = face.bounding_box

        if bbox.bottom_right.x > img_width or bbox.bottom_right.y > img_height:
            raise ValueError("Face bounding box exceeds image dimensions")
        img = img[
            bbox.top_left.y : bbox.bottom_right.y, bbox.top_left.x : bbox.bottom_right.x
        ]
        img_height = bbox.height
        img_width = bbox.width

        scaling_factor: float = min(
            self._input_shape.height / img_height, self._input_shape.width / img_width
        )
        if scaling_factor < 1:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            height, width = img.shape[:2]

        start_x = (self._input_shape.width - width) // 2
        start_y = (self._input_shape.height - height) // 2
        ret[start_y : start_y + height, start_x : start_x + width] = img
        return ret

    # @abstractmethod
    def base_model(self) -> Any:
        pass

    @property
    def name(self) -> str:
        if (
            self._name is None
            or not isinstance(self._name, str)
            or len(self._name) == 0
        ):
            return "<undefined>"
        return self._name

    @property
    def input_shape(self) -> BoxDimensions:
        assert isinstance(self._input_shape, BoxDimensions)
        return self._input_shape

    @property
    def output_shape(self) -> int:
        assert isinstance(self._output_shape, int)
        return self._output_shape

    @staticmethod
    def get_available_extractors() -> List[str]:

        global available_extractors
        if not "available_extractors" in globals():
            available_extractors = reflection.get_derived_classes(
                package=extractors, base_class=Extractor
            )
        return list(available_extractors.keys())

    @staticmethod
    def default() -> str:
        """
        Get the default face extractor name.

        Returns:
        --------
            A string
        """
        return "VGGFace"

    @staticmethod
    def instance(
        name_or_inst: Optional[Union[str, "Extractor"]] = None,
        singleton: bool = True,
    ) -> "Extractor":
        """
        `Extractor` "lazy" factory method.

        Params:
        -------
            `name_or_inst`: A string representing the name of the extractor to instantiate
              or an instance of a `Extractor` subclass. If None, the default detector will be used

            `singleton (bool)`: If True, the factory will return the same instance for the same name

        Returns:
        --------
            An instance of the `Extractor` subclass matching the given name

        Raises:
        -------
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown
            'ImportError': If the detector instance cannot be instantiated
        """

        if name_or_inst is None:
            name_or_inst = Extractor.default()

        if isinstance(name_or_inst, Extractor):
            return name_or_inst

        if not isinstance(name_or_inst, str):
            raise TypeError(
                f"Invalid 'name' argument type [{type(name_or_inst).__name__}] : expected str"
            )
        if not isinstance(singleton, bool):
            raise TypeError(
                f"Invalid 'singleton' argument type [{type(singleton).__name__}] : expected bool"
            )

        name_or_inst = name_or_inst.lower().strip()
        if len(name_or_inst) == 0:
            name_or_inst = Extractor.default()

        global extractors_instances  # singleton design pattern
        if not "extractors_instances" in globals():
            extractors_instances = {}

        global available_extractors
        if not "available_extractors" in globals():
            available_extractors = reflection.get_derived_classes(
                package=extractors, base_class=Extractor
            )

        if name_or_inst not in available_extractors.keys():
            raise NotImplementedError(f"Unknown extractor [{name_or_inst}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_extractors[name_or_inst]()
                logger.debug(
                    f"Instantiated extractor [{name_or_inst}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name_or_inst not in extractors_instances.keys():
                    extractors_instances[name_or_inst] = available_extractors[
                        name_or_inst
                    ]()
                    logger.debug(
                        f"Instantiated extractor [{name_or_inst}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = extractors_instances[name_or_inst]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on extractor [{name_or_inst}] Error: {ex.args}"
            )
            raise ex

        return instance
