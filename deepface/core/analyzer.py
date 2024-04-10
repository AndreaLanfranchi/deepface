from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import time
import numpy
import cv2

from deepface.core import imgutils
from deepface.core import analyzers
from deepface.core import reflection
from deepface.commons.logger import Logger
from deepface.core.types import BoundingBox, DetectedFace, Point

logger = Logger.get_instance()


# Abstract class all specialized face attribute analyzers must inherit from.
class Analyzer(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes

    @dataclass(frozen=True)
    class Results:
        """
        The results of the analysis of a face attribute.

        Attributes:
        -----------
            `value (str)`: The most relevant value of the attribute
            `weights (Dict[str, float])`: The weights of each attribute value
                evaluated in the analysis. The values are normalized to 100.
        """
        name : str
        value: str
        weights: Dict[str, float]

        def __post_init__(self):
            if not isinstance(self.name, str):
                raise TypeError("name must be a string")
            if 0 == len(self.name.strip()):
                raise ValueError("name cannot be empty")
            if not isinstance(self.value, str):
                raise TypeError("value must be a string")
            if 0 == len(self.value.strip()):
                raise ValueError("value cannot be empty")
            if not isinstance(self.weights, dict):
                raise TypeError("weights must be a dictionary")

            if 0 == len(self.weights):
                raise ValueError("weights cannot be empty")
            for key, value in self.weights.items():
                if not isinstance(key, str):
                    raise TypeError("weights keys must be all strings")
                if not isinstance(value, float):
                    raise TypeError("weights values must be all floats")

    @abstractmethod
    def process(self, img: numpy.ndarray) -> Results:
        """
        Process the given image analyzing the face attribute it was built for.

        Args:
        -----
            `img`: The image to analyze. Must be a valid image.
            It is strongly assumed that the image is already a face image.

        Returns:
        --------
            The results of the analysis of the face attribute as
            a `Results` object.

        Raises:
        -------
            `ValueError`: If the image is invalid or empty
        """

        if not imgutils.is_valid_image(img):
            raise ValueError("Invalid image or empty image")


    def _pad_scale_image(
        self,
        img: numpy.ndarray,
        target_shape: Tuple[int, int],
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> numpy.ndarray:
        """
        Pad and scale an image to a target shape.

        Args:
        -----
            img (numpy.ndarray): the input image
            target_shape (Tuple[int, int]): the target shape (height, width)
            face (Optional[Union[DetectedFace, BoundingBox]], optional): the face to be extracted.
                If None, the whole input image is assumed to be a face. Defaults to None.
            pad_color (Tuple[int, int, int], optional): the color for padding. Defaults to (0, 0, 0).
            pad_mode (str, optional): the padding mode. Defaults to "constant".
            pad_position (str, optional): the padding position. Defaults to "center".

        Returns:
        --------
            numpy.ndarray: the padded and scaled image

        Raises:
        -------
            ValueError: if the image or the bounding box is invalid
            TypeError: if the face argument is not of the expected type
        """

        if not imgutils.is_valid_image(img):
            raise ValueError("Invalid image")

        bbox: Optional[BoundingBox] = None
        img_height: int = img.shape[0]
        img_width: int = img.shape[1]

        if face is None:
            bbox = BoundingBox(Point(0, 0), Point(img_width, img_height))
        elif isinstance(face, BoundingBox):
            bbox = face
        elif isinstance(face, DetectedFace):
            bbox = face.bounding_box
        else:
            what: str = "Invalid face argument type"
            what += (
                " expected [BoundingBox | DetectedFace | None], got " + type(face).__name__
            )
            raise TypeError(what)

        if not isinstance(target_shape, tuple) or len(target_shape) != 2:
            raise ValueError("Invalid target shape")
        if not all(isinstance(x, int) for x in target_shape):
            raise TypeError("Target shape must be a tuple of integers")

        if bbox.bottom_right.x > img_width or bbox.bottom_right.y > img_height:
            raise ValueError("Face bounding box exceeds image dimensions")

        scaling_factor: float = min(
            target_shape[0] / img_height,
            target_shape[1] / img_width,
        )
        dsize = (
            int(round(img_width * scaling_factor)),
            int(round(img_height * scaling_factor)),
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
        return ret

    @property
    def name(self) -> str:
        if (
            self._name is None
            or not isinstance(self._name, str)
            or len(self._name.strip()) == 0
        ):
            return "<undefined>"
        return self._name.strip()

    @staticmethod
    def get_available_attributes() -> List[str]:
        """
        Returns the available face attributes that can be analyzed.
        This corresponds to a list of package names in the `analyzers`
        package.

        Returns:
            A list of lowercase strings

        """
        global available_analyzers
        if not "available_analyzers" in globals():
            available_analyzers = reflection.get_derived_classes(
                package=analyzers, base_class=Analyzer
            )

        return list(available_analyzers.keys())

    @staticmethod
    def instance(
        name_or_inst: Union[str, "Analyzer"],
        singleton: bool = True,
    ) -> "Analyzer":
        """
        `Analyzer` "lazy" factory method.

        Args:
        -----
            `name_or_inst`: A string representing the name of the analyzer to instantiate
              or an instance of a `Analyzer` subclass.

        Return:
        -------
            An instance of the `Analyzer` subclass matching the given name

        Raises:
        -------
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the analyzer name empty
            `NotImplementedError`: If the analyzer name is unknown
            'ImportError': If the analyzer instance cannot be instantiated
        """

        if name_or_inst is None:
            raise ValueError("Invalid 'name_or_inst' argument: None")

        if isinstance(name_or_inst, Analyzer):
            return name_or_inst

        if not isinstance(name_or_inst, str):
            what: str = "Invalid 'name_or_inst' argument type"
            raise TypeError(f"{what} [{type(name_or_inst).__name__}] : expected str")

        if not isinstance(singleton, bool):
            what: str = "Invalid 'singleton' argument type"
            raise TypeError(f"{what} [{type(singleton).__name__}] : expected bool")

        name_or_inst = name_or_inst.lower().strip()
        if len(name_or_inst) == 0:
            raise ValueError("Empty analyzer attribute name")

        global analyzers_instances  # singleton design pattern
        if not "analyzers_instances" in globals():
            analyzers_instances = {}

        global available_analyzers
        if not "available_analyzers" in globals():
            available_analyzers = reflection.get_derived_classes(
                package=analyzers, base_class=Analyzer
            )

        if name_or_inst not in available_analyzers.keys():
            raise NotImplementedError(f"Unknown analyzer attribute [{name_or_inst}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_analyzers[name_or_inst]()
                logger.debug(
                    f"Instantiated analyzer [{name_or_inst}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name_or_inst not in analyzers_instances.keys():
                    analyzers_instances[name_or_inst] = available_analyzers[
                        name_or_inst
                    ]()
                    logger.debug(
                        f"Instantiated analyzer [{name_or_inst}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = analyzers_instances[name_or_inst]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on analyzer [{name_or_inst}] Error: {ex.args}"
            )
            raise ex

        return instance
