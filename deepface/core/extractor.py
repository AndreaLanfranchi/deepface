from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import time
import cv2
import numpy

from deepface.core import extractors
from deepface.core import reflection
from deepface.core import imgutils
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace
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

    def _pre_process(
        self,
        img: numpy.ndarray,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> numpy.ndarray:
        """
        Pre-processes the image before extracting the face.

        Args:
        ----
            `img` (numpy.ndarray): The image to be pre-processed
            `face` (Optional[BoundingBox, DetectedFace]): The face to be extracted.
                If None, the whole input image is assumed to be a face

        Returns:
        -------
            numpy.ndarray: The pre-processed image

        Raises:
        ------
            ValueError: If the image or the bounding box is invalid
            TypeError: If the face argument is not of the expected type
        """
        if not imgutils.is_valid_image(img):
            raise ValueError("Invalid image")

        if isinstance(face, BoundingBox):
            if 0 == face.area:
                raise ValueError("Empty face bounding box")
            if face.bottom_right.x > img.shape[1] or face.bottom_right.y > img.shape[0]:
                raise ValueError("Face bounding box exceeds image dimensions")
            img = img[
                face.top_left.y : face.bottom_right.y,
                face.top_left.x : face.bottom_right.x,
            ]

        elif isinstance(face, DetectedFace):
            if 0 == face.bounding_box.area:
                raise ValueError("Empty face bounding box")
            if (
                face.bounding_box.bottom_right.x > img.shape[1]
                or face.bounding_box.bottom_right.y > img.shape[0]
            ):
                raise ValueError("Face bounding box exceeds image dimensions")
            img = face.crop(img)

        elif face is not None:
            what: str = "Invalid face argument type"
            what += (
                " expected [BoundingBox | DetectedFace | None], got "
                + type(face).__name__
            )
            raise TypeError(what)

        # Here we have the face image
        # We now resize and pad the image to the required input shape
        return self._pad_scale_image(img)

    def _pad_scale_image(
        self,
        img: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Pad and scale an image to a target shape.

        Args:
        -----
            img (numpy.ndarray): the input image. It's assumed is a
            valuid face from a valid image. No cropping is done here.

        Returns:
        --------
            numpy.ndarray: the padded and scaled image

        Raises:
        -------
            ValueError: if the image or the bounding box is invalid
            TypeError: if the face argument is not of the expected type
        """
        if self._input_shape.height == 0 or self._input_shape.width == 0:
            raise ValueError("Invalid input shape")

        target_shape = (self._input_shape.height, self._input_shape.width)
        height, width, *_ = img.shape
        scaling_factor: float = min(
            target_shape[0] / height,
            target_shape[1] / width,
        )
        dsize = (
            int(round(width * scaling_factor)),
            int(round(height * scaling_factor)),
        )

        interpolation = cv2.INTER_AREA if scaling_factor < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, dsize=dsize, interpolation=interpolation)
        height, width, *_ = img.shape

        start_x = (target_shape[1] - width) // 2
        start_y = (target_shape[0] - height) // 2
        ret = numpy.zeros(
            shape=(target_shape[0], target_shape[1], 3),
            dtype=numpy.uint8,
        )
        ret[start_y : start_y + height, start_x : start_x + width] = img
        if ret.ndim == 2:
            ret = numpy.stack((ret,) * 3, axis=-1)
        if ret.ndim == 3:
            ret = numpy.expand_dims(ret, axis=0)
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
        name_or_inst: Union[str, "Extractor"] = "default",
        singleton: bool = True,
    ) -> "Extractor":
        """
        `Extractor` "lazy" factory method.

        Params:
        -------
            `name_or_inst`: A string representing the name of the extractor to instantiate
              or an instance of a `Extractor` subclass. When a `Extractor` instance is given,
              the same instance is returned. When a string is given this cases are handled:
              - If the string equals `default`, the default extractor is assumed
              (see `Extractor.default()` method) and returned
              - If the string is a known extractor name, an instance of the corresponding
              extractor is returned

            `singleton (bool)`: If True, the factory will return a singleton instance of the
              extractor. If False, a new instance is returned every time the factory is called

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

        if isinstance(name_or_inst, Extractor):
            return name_or_inst

        if not isinstance(name_or_inst, str):
            what: str = "Invalid 'name_or_inst' argument type. Expected str "
            what += f"got {type(name_or_inst).__name__}"
            raise TypeError(what)

        name_or_inst = name_or_inst.lower().strip()

        if len(name_or_inst) == 0:
            what: str = "Invalid 'name_or_inst' argument value."
            what += " Expected a valid extractor name or `default`. Got empty string"
            raise ValueError(what)

        if name_or_inst == "default":
            name_or_inst = Extractor.default().lower().strip()

        if not isinstance(singleton, bool):
            raise TypeError(
                f"Invalid 'singleton' argument type [{type(singleton).__name__}] : expected bool"
            )

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
