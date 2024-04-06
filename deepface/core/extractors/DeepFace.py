from typing import List, Optional, Union

import os
import zipfile
import tensorflow
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import InsufficentVersionError
from deepface.core.extractor import Extractor as ExtractorBase
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionError("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Convolution2D,
    LocallyConnected2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()


# -------------------------------------
# pylint: disable=too-few-public-methods
class Extractor(ExtractorBase):

    _model: Model

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(152, 152)
        self._output_shape = 4096
        self._initialize()

    def process(
        self,
        img: numpy.ndarray,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> List[float]:

        super().process(img, face)
        img = self._to_required_shape(img, face)
        img = numpy.expand_dims(img, axis=0)
        ret = self._model(img, training=False).numpy()[0].tolist()
        assert len(ret) == self._output_shape
        return ret

    def _initialize(self):

        # Download weights
        file_name = "VGGFace2_DeepFace_weights_val-0.9034.h5"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            compressed_file = f"{file_name}.zip"
            url = "https://github.com/swghosh/DeepFace/releases/download/"
            url += f"weights-vggface2-2d-aligned/{compressed_file}"

            output = os.path.join(folder_utils.get_weights_dir(), compressed_file)
            gdown.download(url, output, quiet=False)

            # unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip
            with zipfile.ZipFile(output, "r") as zip_ref:
                zip_ref.extractall(folder_utils.get_weights_dir())
            os.remove(output)

        # # Build model
        # wt_init = keras.initializers.RandomNormal(mean=0, stddev=0.01)
        # bias_init = keras.initializers.Constant(value=0.5)

        # """
        # Construct certain functions
        # for using some common parameters
        # with network layers
        # """

        # def conv2d_layer(**args):
        #     return keras.layers.Conv2D(
        #         **args,
        #         kernel_initializer=wt_init,
        #         bias_initializer=bias_init,
        #         activation=relu,
        #     )

        # def lc2d_layer(**args):
        #     return keras.layers.LocallyConnected2D(
        #         **args,
        #         kernel_initializer=wt_init,
        #         bias_initializer=bias_init,
        #         activation=relu,
        #     )

        # def dense_layer(**args):
        #     return keras.layers.Dense(
        #         **args, kernel_initializer=wt_init, bias_initializer=bias_init
        #     )

        # model = Sequential([
        #     InputLayer(input_shape=(152, 152, 3), name="I0"),
        #     conv2d_layer(filters=32, kernel_size=(11, 11), name="C1"),
        #     MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"),
        #     conv2d_layer(filters=16, kernel_size=(9, 9), name="C3"),
        #     lc2d_layer(filters=16, kernel_size=(9, 9), name="L4"),
        #     lc2d_layer(filters=16, kernel_size=(7, 7), strides=2, name="L5"),
        #     lc2d_layer(filters=16, kernel_size=(5, 5), name="L6"),
        #     Flatten(name="F0"),
        #     dense_layer(units=4096, activation=relu, name="F7"),
        #     Dropout(rate=0.5, name="D0"),
        #     dense_layer(units=8631, activation="softmax", name="F8"),
        # ], name="DeepFace")

        # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # return model

        base_model = Sequential(
            [
                Convolution2D(
                    32,
                    (11, 11),
                    activation="relu",
                    name="C1",
                    input_shape=(self._input_shape.width, self._input_shape.height, 3),
                ),
                MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"),
                Convolution2D(16, (9, 9), activation=relu, name="C3"),
                LocallyConnected2D(16, (9, 9), activation=relu, name="L4"),
                LocallyConnected2D(16, (7, 7), strides=2, activation=relu, name="L5"),
                LocallyConnected2D(16, (5, 5), activation=relu, name="L6"),
                Flatten(name="F0"),
                Dense(self._output_shape, activation=relu, name="F7"),
                Dropout(rate=0.5, name="D0"),
                Dense(8631, activation="softmax", name="F8"),
            ]
        )

        # base_model.add(
        #     Convolution2D(
        #         32, (11, 11), activation="relu", name="C1", input_shape=(152, 152, 3)
        #     )
        # )
        # base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
        # base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
        # base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
        # base_model.add(
        #     LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5")
        # )
        # base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
        # base_model.add(Flatten(name="F0"))
        # base_model.add(Dense(4096, activation="relu", name="F7"))
        # base_model.add(Dropout(rate=0.5, name="D0"))
        # base_model.add(Dense(8631, activation="softmax", name="F8"))
        # base_model.load_weights(weight_file)

        # drop F8 and D0. F7 is the representation layer.
        self._model = Model(
            inputs=base_model.layers[0].input,
            outputs=base_model.layers[-3].output,
        )

        # return deepface_model
