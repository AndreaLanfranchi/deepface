from abc import ABC, abstractmethod
import time
from typing import Any, List, Optional, Union
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


# Abstract class all specialized face decomposers must inherit from.
# Creates the synthetic digital representation of a face.
# It is assumed the input picture is already a face previously detected.
class Representer(ABC):

    _model: Optional[Union[Model, Sequential]] = None
    _name: Optional[str] = None
    _input_shape: Optional[BoxDimensions] = None
    _output_shape: Optional[int] = None

    @abstractmethod
    def process(self, img: numpy.ndarray) -> List[float]:
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

    @property
    def model(self) -> Union[Model, Sequential]:
        assert isinstance(self._model, (Model, Sequential))
        return self._model

    @staticmethod
    def get_available_decomposers() -> List[str]:
        """
        Get the names of the available face decomposers.

        Returns:
            A list of strings representing the names of the available face decomposers.
        """
        global available_decomposers
        if not "available_decomposers" in globals():
            available_decomposers = reflection.get_derived_classes(
                package=basemodels, base_class=Representer
            )
        return list(available_decomposers.keys())

    @staticmethod
    def get_default() -> str:
        """
        Get the default face decomposer name.

        Returns:
            The name of the default face detector.
        """
        return "VGGFace"

    @staticmethod
    def instance(name: Optional[str] = None, singleton: bool = True) -> "Representer":
        """
        `Decomposer` factory method.

        Args:
            `name (str)`: The name of the detector to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Decomposer` subclass matching the given name

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

        global decomposers_instances  # singleton design pattern
        if not "decomposers_instances" in globals():
            decomposers_instances = {}

        global available_decomposers
        if not "available_decomposers" in globals():
            available_decomposers = reflection.get_derived_classes(
                package=basemodels, base_class=Representer
            )

        if name not in available_decomposers.keys():
            raise NotImplementedError(f"Unknown decomposer [{name}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_decomposers[name]()
                logger.debug(
                    f"Instantiated decomposer [{name}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name not in decomposers_instances.keys():
                    decomposers_instances[name] = available_decomposers[name]()
                    logger.debug(
                        f"Instantiated detector [{name}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = decomposers_instances[name]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on detector [{name}] Error: {ex.args}"
            )
            raise ex

        return instance