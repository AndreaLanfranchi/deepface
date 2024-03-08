from typing import Optional, Union
from abc import ABC, abstractmethod
import numpy
from deepface.commons import package_utils

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model

# Notice that all facial attribute analysis models must be inherited from this class


# pylint: disable=too-few-public-methods
class Demography(ABC):

    _name: Optional[str] = None  # Must be filled by specialized classes
    _model: Model                # The actual model used for the analysis

    @abstractmethod
    def predict(self, img: numpy.ndarray) -> Union[numpy.ndarray, numpy.float64]:
        pass

    @property
    def name(self) -> str:
        return "<undefined>" if self._name is None else self._name
