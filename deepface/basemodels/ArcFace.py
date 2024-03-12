from typing import List

import os
import tensorflow
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import InsufficentVersionRequirement
from deepface.core.representer import Representer as RepresenterBase
from deepface.core.types import BoxDimensions

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionRequirement("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from tensorflow.python.keras.engine import training
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Input,
    Conv2D,
    BatchNormalization,
    PReLU,
    Add,
    Dropout,
    Flatten,
    Dense,
)

# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()

# ArcFace respresenter model
class Representer(RepresenterBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=112, height=112)
        self._output_shape = 512
        self._initialize()

    def process(self, img: numpy.ndarray) -> List[float]:
        # TODO: shouldn't we ensure image is resized to fit in the input_shape?
        return self._model(img, training=False).numpy()[0].tolist()

    def _initialize(self):

        file_name: str = "arcface_weights.h5"
        weight_file: str = os.path.join(folder_utils.get_weights_dir(), file_name)
        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, weight_file, quiet=False)

        self._base_model = self._ResNet34()
        inputs = self._base_model.inputs[0] if self._base_model.inputs else None
        arcface_model = (
            self._base_model.outputs[0] if self._base_model.outputs else None
        )
        arcface_model = BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
        arcface_model = Dropout(0.4)(arcface_model)
        arcface_model = Flatten()(arcface_model)
        arcface_model = Dense(
            units=self.output_shape,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_normal",
        )(arcface_model)
        embedding = BatchNormalization(
            momentum=0.9, epsilon=2e-5, name="embedding", scale=True
        )(arcface_model)

        self._model = Model(inputs, embedding, name=self._base_model.name)
        self._model.load_weights(weight_file)

    def _ResNet34(self) -> Model:
        """
        ResNet34 model
        Returns:
            model (Model)
        """
        img_input = Input(shape=(self._input_shape.height, self._input_shape.height, 3))

        x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
        x = Conv2D(
            64,
            3,
            strides=1,
            use_bias=False,
            kernel_initializer="glorot_normal",
            name="conv1_conv",
        )(x)
        x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
        x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
        x = self._stack_fn(x)

        return training.Model(img_input, x, name="ResNet34")

    def _block1(
        self, x, filters, kernel_size=3, stride=1, conv_shortcut=True, name: str = str()
    ):
        bn_axis = 3

        if conv_shortcut:
            shortcut = Conv2D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                kernel_initializer="glorot_normal",
                name=name + "_0_conv",
            )(x)
            shortcut = BatchNormalization(
                axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_0_bn"
            )(shortcut)
        else:
            shortcut = x

        x = BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn"
        )(x)
        x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
        x = Conv2D(
            filters,
            3,
            strides=1,
            kernel_initializer="glorot_normal",
            use_bias=False,
            name=name + "_1_conv",
        )(x)
        x = BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn"
        )(x)
        x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)

        x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
        x = Conv2D(
            filters,
            kernel_size,
            strides=stride,
            kernel_initializer="glorot_normal",
            use_bias=False,
            name=name + "_2_conv",
        )(x)
        x = BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn"
        )(x)

        x = Add(name=name + "_add")([shortcut, x])
        return x

    def _stack1(self, x, filters, blocks, stride1=2, name: str = str()):
        x = self._block1(x, filters, stride=stride1, name=name + "_block1")
        for i in range(2, blocks + 1):
            x = self._block1(
                x, filters, conv_shortcut=False, name=name + "_block" + str(i)
            )
        return x

    def _stack_fn(self, x):
        x = self._stack1(x, 64, 3, name="conv2")
        x = self._stack1(x, 128, 4, name="conv3")
        x = self._stack1(x, 256, 6, name="conv4")
        return self._stack1(x, 512, 3, name="conv5")
