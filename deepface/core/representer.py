from abc import ABC, abstractmethod
from typing import List, Optional, Union

import time
import cv2
import numpy

from deepface import basemodels
from deepface.commons import package_utils
from deepface.core import reflection
from deepface.core.types import BoxDimensions
from deepface.commons.logger import Logger

logger = Logger.get_instance()

tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
else:
    from keras.models import Model, Sequential


# Abstract class all specialized face representers must inherit from.
# Creates the synthetic digital representation of a face.
# It is assumed the input picture is already a face previously detected.
class Representer(ABC):

    _model: Union[Model, Sequential]
    _name: str
    _input_shape: BoxDimensions
    _output_shape: int

    @abstractmethod
    def process(self, img: numpy.ndarray) -> List[float]:
        pass

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

    @property
    def model(self) -> Union[Model, Sequential]:
        assert isinstance(self._model, (Model, Sequential))
        return self._model

    @staticmethod
    def get_available_representers() -> List[str]:

        global available_representers
        if not "available_representers" in globals():
            available_representers = reflection.get_derived_classes(
                package=basemodels, base_class=Representer
            )
        return list(available_representers.keys())

    @staticmethod
    def get_default() -> str:
        return "VGGFace"

    @staticmethod
    def instance(name: Optional[str] = None, singleton: bool = True) -> "Representer":
        """
        `Representer` factory method.

        Args:
            `name (str)`: The name of the representer to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Representer` subclass matching the given name

        Raises:
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown
            'ImportError': If the detector instance cannot be instantiated
        """
        if name is None:
            name = Representer.get_default()
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
            name = Representer.get_default()

        global representers_instances  # singleton design pattern
        if not "representers_instances" in globals():
            representers_instances = {}

        global available_representers
        if not "available_representers" in globals():
            available_representers = reflection.get_derived_classes(
                package=basemodels, base_class=Representer
            )

        if name not in available_representers.keys():
            raise NotImplementedError(f"Unknown respresenter [{name}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_representers[name]()
                logger.debug(
                    f"Instantiated representer [{name}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name not in representers_instances.keys():
                    representers_instances[name] = available_representers[name]()
                    logger.debug(
                        f"Instantiated representer [{name}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = representers_instances[name]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on representer [{name}] Error: {ex.args}"
            )
            raise ex

        return instance
