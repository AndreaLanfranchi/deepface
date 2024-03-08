import os
import gdown
import numpy
from deepface.basemodels import VGGFace
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.core.analyzer import Analyzer

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
class ApparentAgeClient(Analyzer):
    """
    Age model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Age"

    def predict(self, img: numpy.ndarray) -> numpy.float64:
        age_predictions = self.model.predict(img, verbose=0)[0, :]
        return find_apparent_age(age_predictions)


def load_model() -> Model:
    """
    Construct age model, download its weights and load
    Returns:
        model (Model)
    """

    model = VGGFace.base_model()

    # --------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    file_name = "age_model_weights.h5"
    url = f"https://github.com/serengil/deepface_models/releases/download/v1.0/{file_name}",
    output = os.path.join(folder_utils.get_weights_dir(), file_name)

    if os.path.isfile(output) != True:
        logger.info(f"Download : {file_name}")
        gdown.download(url, output, quiet=False)

    age_model.load_weights(output)

    return age_model

    # --------------------------


def find_apparent_age(age_predictions: numpy.ndarray) -> numpy.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (?)
    Returns:
        apparent_age (float)
    """
    output_indexes = numpy.array(list(range(0, 101)))
    apparent_age = numpy.sum(age_predictions * output_indexes)
    return apparent_age
