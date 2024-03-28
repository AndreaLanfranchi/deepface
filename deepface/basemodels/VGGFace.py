from typing import List

import os
import tensorflow
import gdown
import numpy

from deepface.core.types import BoxDimensions
from deepface.modules import verification
from deepface.core.extractor import Extractor as ExtractorBase
from deepface.core.exceptions import InsufficentVersionRequirement
from deepface.commons import folder_utils
from deepface.commons.logger import Logger


tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionRequirement("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from keras.models import Model, Sequential
from keras.layers import (
    Convolution2D,
    ZeroPadding2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Activation,
)
# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()


class Extractor(ExtractorBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(224, 224)
        self._output_shape = 4096
        self._initialize()

    def process(self, img: numpy.ndarray) -> List[float]:
        super().process(img)
        img = self.to_required_shape(img)
        embedding = self._model(img, training=False).numpy()[0].tolist()
        embedding = verification.l2_normalize(embedding)
        return embedding.tolist()

    def _initialize(self):

        file_name = "vgg_face_weights.h5"
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")
            url = "https://github.com/serengil/deepface_models/"
            url += f"releases/download/v1.0/{file_name}"
            gdown.download(url, output, quiet=False)

        base_model = self.base_model()
        base_model.load_weights(output)

        # 2622d dimensional model
        # vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

        # 4096 dimensional model offers 6% to 14% increasement on accuracy!
        # - softmax causes underfitting
        # - added normalization layer to avoid underfitting with euclidean
        # as described here: https://github.com/serengil/deepface/issues/944
        base_model_output = Sequential()
        base_model_output = Flatten()(base_model.layers[-5].output)
        # keras backend's l2 normalization layer troubles some gpu users (e.g. issue 957, 966)
        # base_model_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(
        #     base_model_output
        # )
        self._model = Model(inputs=base_model.input, outputs=base_model_output)

    def base_model(self) -> Sequential:
        ret = Sequential(
            [
                ZeroPadding2D(
                    (1, 1),
                    input_shape=(self.input_shape.width, self.input_shape.height, 3),
                ),
                Convolution2D(64, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2), strides=(2, 2)),
                ZeroPadding2D((1, 1)),
                Convolution2D(128, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2), strides=(2, 2)),
                ZeroPadding2D((1, 1)),
                Convolution2D(256, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(256, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(256, (3, 3), activation="relu"),
                MaxPooling2D((2, 2), strides=(2, 2)),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, (3, 3), activation="relu"),
                MaxPooling2D((2, 2), strides=(2, 2)),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, (3, 3), activation="relu"),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, (3, 3), activation="relu"),
                MaxPooling2D((2, 2), strides=(2, 2)),
                Convolution2D(self._output_shape, (7, 7), activation="relu"),
                Dropout(0.5),
                Convolution2D(self._output_shape, (1, 1), activation="relu"),
                Dropout(0.5),
                Convolution2D(2622, (1, 1)),
                Flatten(),
                Activation("softmax"),
            ]
        )
        return ret
