from typing import List, Optional

import os
import tensorflow
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.types import BoxDimensions
from deepface.core.exceptions import InsufficentVersionRequirement
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


# FaceNet-128d model
class Extractor(ExtractorBase):

    _model: Model

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=160, height=160)
        self._output_shape = int(128)
        self._initialize()

    def _initialize(self):
        file_name = "facenet_weights.h5"
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
        img = self.to_required_shape(img)
        return self._model(img, training=False).numpy()[0].tolist()


def _scaling(x, scale):
    return x * scale


def _conv2d(
    inp,
    filters: int,
    kernel_size: list[int],
    strides: int = int(1),
    padding: str = "same",
    use_bias: bool = False,
    activation: Optional[str] = "relu",
):
    ret = Conv2D(filters, kernel_size, strides, padding=padding, use_bias=use_bias)(inp)
    ret = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False)(ret)
    if activation != "None":
        ret = Activation(activation)(ret)
    return ret


def _stem_block(inp):
    ret = _conv2d(inp, 32, [3], strides=2, padding="valid")
    ret = _conv2d(ret, 32, [3], padding="valid")
    ret = _conv2d(ret, 64, [3])

    ret = MaxPooling2D(3, strides=2)(ret)

    ret = _conv2d(ret, 80, [1], padding="valid")
    ret = _conv2d(ret, 192, [3], padding="valid")
    ret = _conv2d(ret, 256, [3], strides=2, padding="valid")

    return ret


def _inception_resnet_a_block(inp):
    branch_0 = _conv2d(inp, 32, [1])
    branch_1 = _conv2d(inp, 32, [1])
    branch_1 = _conv2d(branch_1, 32, [3])
    branch_2 = _conv2d(inp, 32, [1])
    branch_2 = _conv2d(branch_2, 32, [3])
    branch_2 = _conv2d(branch_2, 32, [3])

    mixed = Concatenate(axis=3)([branch_0, branch_1, branch_2])
    up = Conv2D(256, 1, strides=1, padding="same", use_bias=True)(mixed)
    up = Lambda(_scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(
        up
    )
    ret = add([inp, up])
    return Activation("relu")(ret)


def _reduction_a_block(inp):
    branch_0 = _conv2d(inp, 384, [3], strides=2, padding="valid")
    branch_1 = _conv2d(inp, 192, [1])
    branch_1 = _conv2d(branch_1, 192, [3])
    branch_1 = _conv2d(branch_1, 256, [3], strides=2, padding="valid")
    branch_2 = MaxPooling2D(3, strides=2, padding="valid")(inp)
    return Concatenate(axis=3)([branch_0, branch_1, branch_2])


def _inception_resnet_b_block(inp):
    branch_0 = _conv2d(inp, 128, [1])
    branch_1 = _conv2d(inp, 128, [1])
    branch_1 = _conv2d(branch_1, 128, [1, 7])

    mixed = Concatenate(axis=3)([branch_0, branch_1])
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True)(mixed)
    up = Lambda(_scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(
        up
    )
    ret = add([inp, up])
    return Activation("relu")(ret)


def _reduction_b_block(inp):
    branch_0 = _conv2d(inp, 256, [1])
    branch_0 = _conv2d(branch_0, 384, [3], strides=2, padding="valid")
    branch_1 = _conv2d(inp, 256, [1])
    branch_1 = _conv2d(branch_1, 256, [3], strides=2, padding="valid")
    branch_2 = _conv2d(inp, 256, [1])
    branch_2 = _conv2d(branch_2, 256, [3])
    branch_2 = _conv2d(branch_2, 256, [3], strides=2, padding="valid")
    branch_3 = MaxPooling2D(3, strides=2, padding="valid")(inp)
    return Concatenate(axis=3)([branch_0, branch_1, branch_2, branch_3])


def _inception_resnet_c_block(inp, activation="relu"):
    branch_0 = _conv2d(inp, 192, [1])
    branch_1 = _conv2d(inp, 192, [1])
    branch_1 = _conv2d(branch_1, 192, [1, 3])
    branch_1 = _conv2d(branch_1, 192, [3, 1])

    mixed = Concatenate(axis=3)([branch_0, branch_1])
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True)(mixed)
    up = Lambda(_scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(
        up
    )
    ret = add([inp, up])
    if activation != "None":
        return Activation(activation)(ret)
    return ret


def InceptionResNetV1(input_shape: BoxDimensions, output: int = 128) -> Model:
    inp = Input(shape=(input_shape.height, input_shape.width, 3))
    x = _stem_block(inp)

    for _ in range(5):
        x = _inception_resnet_a_block(x)

    x = _reduction_a_block(x)

    for _ in range(10):
        x = _inception_resnet_b_block(x)

    x = _reduction_b_block(x)

    for _ in range(5):
        x = _inception_resnet_c_block(x)

    x = _inception_resnet_c_block(x, activation="None")

    # Classification block
    x = GlobalAveragePooling2D()(x)
    x = Dropout(1.0 - 0.8)(x)
    # Bottleneck
    x = Dense(output, use_bias=False)(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False)(x)

    # Create model
    return Model(inp, x, name="inception_resnet_v1")
