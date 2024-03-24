from abc import ABC, abstractmethod
from typing import List, Optional

import time
import cv2
import numpy

from deepface import basemodels
from deepface.core import reflection
from deepface.core.types import BoxDimensions
from deepface.commons.logger import Logger

logger = Logger.get_instance()


# Abstract class all specialized face extractors must inherit from.
# Creates the synthetic digital representation of a face.
# It is assumed the input picture is already a face previously detected.
class Extractor(ABC):
    """
    Interface for digital face(s) representation.
    """

    _name: str
    _input_shape: BoxDimensions
    _output_shape: int

    @abstractmethod
    def process(self, img: numpy.ndarray) -> List[float]:
        """
        Extracts the digital representation of a face from an image.
        Note: It is strongly assumed that the input image is already a face
        previously detected and cropped.
        """
        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array")


    def _scale_pad(self, img: numpy.ndarray) -> numpy.ndarray:
        """
        Scales and pads the image to fit the input shape of the model.
        Also validates the image shape.

        Args:
            img (numpy.ndarray): The image to be scaled and padded

        Returns:
            numpy.ndarray: The scaled and padded image

        Raises:
            ValueError: If the image shape is invalid
        """

        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)

        if len(img.shape) != 3:
            raise ValueError("Invalid image shape")

        (height, width) = img.shape[:2]
        if height * width == 0:
            raise ValueError("Invalid image dimensions")

        img = cv2.resize(img, (self._input_shape.width, self._input_shape.height))
        img = numpy.expand_dims(img, axis=0)
        # when called from verify, this is already normalized. But needed when user given.
        if img.max() > 1:
            img = (img.astype(numpy.float32) / 255.0).astype(numpy.float32)

        if (
            img.shape[0] != self.input_shape.height
            or img.shape[1] != self.input_shape.width
        ):
            img = cv2.resize(img, (self.input_shape.width, self.input_shape.height))
        return img

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
                package=basemodels, base_class=Extractor
            )
        return list(available_extractors.keys())

    @staticmethod
    def get_default() -> str:
        return "VGGFace"

    @staticmethod
    def instance(name: Optional[str] = None, singleton: bool = True) -> "Extractor":
        """
        `Representer` factory method.

        Args:
            `name (str)`: The name of the representer to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Extractor` subclass matching the given name

        Raises:
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown
            'ImportError': If the detector instance cannot be instantiated
        """
        if name is None:
            name = Extractor.get_default()
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
            name = Extractor.get_default()

        global extractors_instances  # singleton design pattern
        if not "extractors_instances" in globals():
            extractors_instances = {}

        global available_extractors
        if not "available_extractors" in globals():
            available_extractors = reflection.get_derived_classes(
                package=basemodels, base_class=Extractor
            )

        if name not in available_extractors.keys():
            raise NotImplementedError(f"Unknown extractor [{name}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_extractors[name]()
                logger.debug(
                    f"Instantiated extractor [{name}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name not in extractors_instances.keys():
                    extractors_instances[name] = available_extractors[name]()
                    logger.debug(
                        f"Instantiated extractor [{name}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = extractors_instances[name]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on extractor [{name}] Error: {ex.args}"
            )
            raise ex

        return instance
