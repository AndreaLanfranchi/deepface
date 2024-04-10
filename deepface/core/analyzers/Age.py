import os
from typing import Optional, Union
import tensorflow
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core import imgutils
from deepface.core.analyzer import Analyzer as AnalyzerBase
from deepface.core.exceptions import InsufficentVersionError
from deepface.core.extractor import Extractor
from deepface.core.types import BoundingBox, DetectedFace

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionError("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Activation

# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()


# pylint: disable=too-few-public-methods
class Analyzer(AnalyzerBase):

    _model: Model  # The actual model used for the analysis
    _classes: int  # The number of classes the model can predict
    _output_indexes: numpy.ndarray  # The indexes of the output values of the model

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._classes = 101
        self._output_indexes = numpy.array(list(range(0, self._classes)))
        self.__initialize()

    def process(
        self,
        img: numpy.ndarray,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> AnalyzerBase.Results:

        img = self._pre_process(img,face)
        
        estimates = self._model.predict(img, verbose=0)[0, :]
        attribute_name = self.name.lower()
        attribute_value = str(
            round(numpy.sum(estimates * self._output_indexes).astype(float))
        )
        weights = {str(i): float(estimates[i]) for i in range(0, self._classes) if estimates[i] != float(0)}
        return AnalyzerBase.Results(attribute_name, attribute_value, weights)

    def __initialize(self):

        file_name = "age_model_weights.h5"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)
        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, weight_file, quiet=False)

        base_model: Sequential = Extractor.instance().base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(self._classes, (1, 1), name="predictions")(
            base_model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        self._model = Model(inputs=base_model.input, outputs=base_model_output)
        self._model.load_weights(weight_file)
