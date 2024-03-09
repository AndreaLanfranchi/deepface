from abc import ABC, abstractmethod
import time
from typing import Any, List, Optional, Tuple, Union
import numpy

from deepface import basemodels
from deepface.commons import package_utils
from deepface.core import reflection
from deepface.commons.logger import Logger

logger = Logger.get_instance()

tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# Abstract class all specialized face decomposers must inherit from.
class Decomposer(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes

    model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

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
                package=basemodels, base_class=Decomposer
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
    def instance(name: Optional[str] = None, singleton: bool = True) -> "Decomposer":
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
            name = Decomposer.get_default()
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
            name = Decomposer.get_default()

        global decomposers_instances  # singleton design pattern
        if not "decomposers_instances" in globals():
            decomposers_instances = {}

        global available_decomposers
        if not "available_decomposers" in globals():
            available_decomposers = reflection.get_derived_classes(
                package=basemodels, base_class=Decomposer
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
