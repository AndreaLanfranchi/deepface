from typing import List

import os
import tensorflow
import gdown
import numpy

from deepface.basemodels.Facenet128 import InceptionResNetV1
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import InsufficentVersionRequirement
from deepface.core.types import BoxDimensions
from deepface.core.extractor import Extractor as ExtractorBase

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionRequirement("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    add,
)
from tensorflow.keras import backend as K
# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()

class Extractor(ExtractorBase):

    _model: Model

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=160, height=160)
        self._output_shape = int(512)
        self._initialize()

    def _initialize(self):
        file_name = "facenet512_weights.h5"
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")
            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, output, quiet=False)

        self._model = InceptionResNetV1(self._input_shape, self._output_shape)
        self._model.load_weights(output)

    def process(self, img: numpy.ndarray) -> List[float]:

        super().process(img)
        # TODO: shouldn't we ensure image is resized to fit in the input_shape?
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        return self._model(img, training=False).numpy()[0].tolist()
