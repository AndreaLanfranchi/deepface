from typing import Optional, Union
from abc import ABC, abstractmethod
import numpy
from deepface.commons import package_utils


# Notice that all facial attribute analysis models must be inherited from this class
class Demography(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes

    @abstractmethod
    def predict(self, img: numpy.ndarray) -> Union[numpy.ndarray, numpy.float64]:
        pass

    @property
    def name(self) -> str:
        return "<undefined>" if self._name is None else self._name
