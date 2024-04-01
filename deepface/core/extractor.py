from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import time
import cv2
import numpy

from deepface.core import extractors
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

        if img.shape[0] * img.shape[1] == 0:
            raise ValueError("Invalid image dimensions")

    def to_required_shape(self, img: numpy.ndarray) -> numpy.ndarray:
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
        height, width = img.shape[:2]
        ret = numpy.zeros(
            shape=(self._input_shape.height, self._input_shape.width, 3),
            dtype=numpy.uint8,
        )

        scaling_factor: float = min(
            self._input_shape.height / height, self._input_shape.width / width
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
    def get_default() -> str:
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
              
            `singleton (bool)`: If True, the same instance will be returned

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
            name_or_inst = Extractor.get_default()

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
            name_or_inst = Extractor.get_default()

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
                    extractors_instances[name_or_inst] = available_extractors[name_or_inst]()
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
