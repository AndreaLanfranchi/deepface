from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
import time
import numpy

from deepface.core import analyzers
from deepface.core import reflection
from deepface.commons.logger import Logger

logger = Logger.get_instance()


# Abstract class all specialized face attribute analyzers must inherit from.
class Analyzer(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes

    @abstractmethod
    def process(
        self, img: numpy.ndarray, detail: bool = False
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Process the given image analyzing the face attribute.

        Returns:
            The result of the analysis is returned in the form of
            Dict[str, Union[str, Dict[str, float]]] object.

            For example:
            {
                "attribute": "value"
            }
            where the literal "attribute" is the name of the attribute
            being analyzed (lowercase) and value is the most relevant
            or if you prefer "dominant" attribute value.

            If the `detail` parameter is set to True, the result will also
            contain more detailed information about the analysis providing
            the waights (notmalized to 100) of each attribute value
            evaluated. For example (in case of emotion analysis):
            {
                "emotion": "happiness",
                "emotion_analysis": {
                    "happiness": "50.0",
                    "sadness": "10.0",
                    "anger": 5.0,
                    [...]
                }
            }

            Note ! The "detail" key might not cause the return of the
            detailed information in some cases (for example, if the
            attribute 'age').

        """

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
    def instance(name: str, singleton: bool = True) -> "Analyzer":
        """
        `Analyzer` factory method.

        Args:
            `name (str)`: The name of the analyzer to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Demography` subclass matching the given name

        Raises:
            `TypeError`: If the name or singleton arguments are not of the expected type
            `ValueError`: If the analyzer name empty
            `NotImplementedError`: If the analyzer name is unknown
            'ImportError': If the analyzer instance cannot be instantiated
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Invalid 'name' argument type [{type(name).__name__}] : expected str"
            )
        if not isinstance(singleton, bool):
            raise TypeError(
                f"Invalid 'singleton' argument type [{type(singleton).__name__}] : expected bool"
            )

        name = name.lower().strip()
        if len(name) == 0:
            raise ValueError("Empty analyzer name")

        global analyzers_instances  # singleton design pattern
        if not "analyzers_instances" in globals():
            analyzers_instances = {}

        global available_analyzers
        if not "available_analyzers" in globals():
            available_analyzers = reflection.get_derived_classes(
                package=analyzers, base_class=Analyzer
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
