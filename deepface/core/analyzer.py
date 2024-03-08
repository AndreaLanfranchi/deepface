from typing import Optional, Union
from abc import ABC, abstractmethod

import time
import numpy

from deepface.commons import package_utils
from deepface.core import reflection, analyzers
from deepface.commons.logger import Logger
logger = Logger()

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model


# Abstract class all specialized face attribute analyzers must inherit from.
# A face detection consists in finding [0,inf) faces in an image and
# returning the region of the image where the face is located.
class Analyzer(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes
    _model: Model  # The actual model used for the analysis

    @abstractmethod
    def process(self, img: numpy.ndarray) -> Union[numpy.ndarray, numpy.float64]:
        pass

    @staticmethod
    def instance(name: str, singleton: bool = True) -> "Analyzer":
        """
        `Analyzer` factory method.

        Args:
            `name (str)`: The name of the analyzer to instantiate
            `singleton (bool)`: If True, the same instance will be returned

        Return:
            An instance of the `Analyzer` subclass matching the given name

        Raises:
            `ValueError`: If the analyzer name empty
            `NotImplementedError`: If the analyzer name is unknown

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
                package=analyzers, base_class=Analyzer
            )

        if name not in available_analyzers.keys():
            raise NotImplementedError(f"Unknown detector : {name}")

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
                instance = analyzers_instances[name]
                logger.debug(
                    f"Instantiated analyzer [{name}] ({time.time() - tic:.3f} seconds)"
                )
        except Exception as ex:
            logger.critical(f"{type(ex).__name__} on analyzer [{name}] Error: {ex.args}")
            raise ex

        return instance
