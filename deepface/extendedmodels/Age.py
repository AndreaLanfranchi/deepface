import os
import gdown
import numpy
from deepface.basemodels import VGGFace
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.models.Demography import Demography

logger = Logger(module="extendedmodels.Age")

# ----------------------------------------
# dependency configurations

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

# ----------------------------------------


# pylint: disable=too-few-public-methods
class ApparentAgeClient(Demography):

    _model: Model  # The actual model used for the analysis

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def predict(self, img: numpy.ndarray) -> numpy.float64:
        age_predictions = self.model.predict(img, verbose=0)[0, :]
        apparent_age = numpy.sum(age_predictions * self._output_indexes)
        return apparent_age

    def __initialize(self):

        classes = 101  # TDOO: What is this magic number?
        self._output_indexes = numpy.array(list(range(0, classes)))

        with VGGFace.base_model() as base_model:
            base_model_output = Sequential()
            base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
                base_model.layers[-4].output
            )
            base_model_output = Flatten()(base_model_output)
            base_model_output = Activation("softmax")(base_model_output)
            self._model = Model(inputs=base_model.input, outputs=base_model_output)

        file_name = "age_model_weights.h5"
        url = (
            f"https://github.com/serengil/deepface_models/releases/download/v1.0/{file_name}",
        )
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")
            gdown.download(url, output, quiet=False)

        self._model.load_weights(output)
