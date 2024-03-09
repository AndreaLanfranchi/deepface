from typing import Dict, Union
import os
import gdown
import numpy
from deepface.basemodels import VGGFace
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.core.analyzer import Analyzer as AnalyzerBase

logger = Logger.get_instance()

# --------------------------
# pylint: disable=line-too-long
# --------------------------
# dependency configurations
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation


# pylint: disable=too-few-public-methods
class Analyzer(AnalyzerBase):

    _model: Model  # The actual model used for the analysis
    _labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def process(
        self, img: numpy.ndarray, detail: bool = False
    ) -> Dict[str, Union[str, Dict[str, float]]]:

        result = {}
        attribute = self.name.lower()

        estimates = self._model.predict(img, verbose=0)[0, :]
        result[attribute] = self._labels[numpy.argmax(estimates)]

        if detail:
            details = {}
            estimates_sum = numpy.sum(estimates)
            for i, label in enumerate(self._labels):
                estimate = round(estimates[i] * 100 / estimates_sum, 2)
                details[label] = estimate
            result[f"{attribute}_analysis"] = details

        return result

    def __initialize(self):

        classes = 6  # TDOO: What is this magic number?
        base_model = VGGFace.base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
            base_model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        self._model = Model(inputs=base_model.input, outputs=base_model_output)

        file_name = "race_model_single_batch.h5"
        url = f"https://github.com/serengil/deepface_models/releases/download/v1.0/{file_name}"
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")
            gdown.download(url, output, quiet=False)

        self._model.load_weights(output)