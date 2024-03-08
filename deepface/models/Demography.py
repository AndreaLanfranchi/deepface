from typing import Optional, Union
from abc import ABC, abstractmethod
import time
import numpy

from deepface import extendedmodels
from deepface.core import reflection
from deepface.commons.logger import Logger

logger = Logger()


# Notice that all facial attribute analysis models must be inherited from this class
class Demography(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes

    @abstractmethod
    def predict(self, img: numpy.ndarray) -> Union[numpy.ndarray, numpy.float64]:
        pass

    @property
    def name(self) -> str:
        return "<undefined>" if self._name is None else self._name

    @staticmethod
    def instance(name: str, singleton: bool = True) -> "Demography":
        """
        `Demography` factory method.

        Args:
            `name (str)`: The name of the analyzer to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Demography` subclass matching the given name

        Raises:
            `ValueError`: If the detector name empty
            `NotImplementedError`: If the detector name is unknown

        """
        name = name.lower().strip()
        if len(name) == 0:
            raise ValueError("Empty analyzer name")

        global analyzers_instances  # singleton design pattern
        if not "analyzers_instances" in globals():
            analyzers_instances = {}

        global available_analyzers
        if not "available_analyzers" in globals():
            available_analyzers = reflection.get_derived_classes(
                package=extendedmodels, base_class=Demography
            )

        if name not in available_analyzers.keys():
            raise NotImplementedError(f"Unknown analyzer [{name}]")

        tic = time.time()
        try:
            if not singleton:
                instance = available_analyzers[name]()
                logger.debug(
                    f"Instantiated analyzer [{name}] ({time.time() - tic:.3f} seconds)"
                )
            else:
                if name not in analyzers_instances.keys():
                    analyzers_instances[name] = available_analyzers[name]()
                    logger.debug(
                        f"Instantiated analyzer [{name}] ({time.time() - tic:.3f} seconds)"
                    )
                instance = analyzers_instances[name]
        except Exception as ex:
            logger.critical(
                f"{type(ex).__name__} on analyzer [{name}] Error: {ex.args}"
            )
            raise ex

        return instance
