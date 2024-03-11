from typing import List

import os
import tensorflow
import gdown
import numpy

from deepface.core.exceptions import InsufficentVersionRequirement
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.representer import Representer as RepresenterBase
from deepface.core.types import BoxDimensions

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionRequirement("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    Input,
    Add,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()

# DeepID respresenter model
class Representer(RepresenterBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=47, height=55)
        self._output_shape = 160
        self._initialize()

    def process(self, img: numpy.ndarray) -> List[float]:
        return self._model(img, training=False).numpy()[0].tolist()

    def _initialize(self):

        file_name: str = "deepid_keras_weights.h5"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)
        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, weight_file, quiet=False)

        myInput = Input(shape=(self._input_shape.height, self._input_shape.width, 3))

        x = Conv2D(
            20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3)
        )(myInput)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
        x = Dropout(rate=0.99, name="D1")(x)

        x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
        x = Dropout(rate=0.99, name="D2")(x)

        x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
        x = Dropout(rate=0.99, name="D3")(x)

        x1 = Flatten()(x)
        fc11 = Dense(160, name="fc11")(x1)

        x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
        x2 = Flatten()(x2)
        fc12 = Dense(160, name="fc12")(x2)

        y = Add()([fc11, fc12])
        y = Activation("relu", name="deepid")(y)

        self._model = Model(inputs=[myInput], outputs=y)
        self._model.load_weights(weight_file)
