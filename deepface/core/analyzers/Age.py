from typing import Dict, Union

import os
import tensorflow
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.analyzer import Analyzer as AnalyzerBase
from deepface.core.exceptions import InsufficentVersionRequirement
from deepface.core.extractor import Extractor

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionRequirement("Tensorflow reequires version >=2.0.0")

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

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def process(
        self, img: numpy.ndarray, detail: bool = False
    ) -> Dict[str, Union[str, Dict[str, float]]]:

        result = {}
        attribute = self.name.lower()
        estimates = self._model.predict(img, verbose=0)[0, :]
        attribute_value = numpy.sum(estimates * self._output_indexes).astype(float)
        result[attribute] = attribute_value

        if detail:
            details = {}
            result[f"{attribute}_analysis"] = details

        return result

    def __initialize(self):

        file_name = "age_model_weights.h5"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)
        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, weight_file, quiet=False)

        classes = 101  # TDOO: What is this magic number?
        self._output_indexes = numpy.array(list(range(0, classes)))

        base_model: Sequential = Extractor.instance().base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
            base_model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        self._model = Model(inputs=base_model.input, outputs=base_model_output)
        self._model.load_weights(weight_file)
