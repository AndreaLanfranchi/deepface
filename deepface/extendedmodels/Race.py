import os
import gdown
import numpy
from deepface.basemodels import VGGFace
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.core.analyzer import Analyzer

logger = Logger(module="extendedmodels.Race")

# --------------------------
# pylint: disable=line-too-long
# --------------------------
# dependency configurations
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation
# --------------------------
# Labels for the ethnic phenotypes that can be detected by the model.
labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]


# pylint: disable=too-few-public-methods
class RaceClient(Analyzer):

    _model: Model  # The actual model used for the analysis

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def predict(self, img: numpy.ndarray) -> numpy.ndarray:
        return self._model.predict(img, verbose=0)[0, :]

    def __initialize(self):

        classes = 6  # TDOO: What is this magic number?
        with VGGFace.base_model() as base_model:
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
