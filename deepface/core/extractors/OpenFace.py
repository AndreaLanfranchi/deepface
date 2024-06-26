from typing import List, Optional, Union

import os
import gdown
import tensorflow as tf
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import InsufficentVersionError
from deepface.core.extractor import Extractor as ExtractorBase
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace


tensorflow_version_major = int(tf.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionError("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
import keras
from keras.models import Model
from keras.layers import (
    Conv2D,
    ZeroPadding2D,
    Input,
    concatenate,
    Dense,
    Activation,
    Lambda,
    Flatten,
    BatchNormalization,
    MaxPooling2D,
    AveragePooling2D,
)
from keras import backend as K

# pylint: enable=wrong-import-position
# pylint: enable=wrong-import-order

logger = Logger.get_instance()


# pylint: disable=too-few-public-methods
class Extractor(ExtractorBase):

    _model: Model

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=96, height=96)
        self._output_shape = 128
        self._initialize()

    def _initialize(self):

        file_name = "openface_weights.h5"
        output = os.path.join(folder_utils.get_weights_dir(), file_name)
        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")
            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, output, quiet=False)

        self._model = _build_model(self._input_shape, self._output_shape)
        self._model.load_weights(output)

    def process(
        self,
        img: numpy.ndarray,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> List[float]:

        img = self._pre_process(img, face)
        ret = self._model(img, training=False).numpy()[0].tolist()
        assert len(ret) == self._output_shape
        return ret


def _build_model(input_shape: BoxDimensions, output_shape: int) -> Model:

    #TODO : This is way too long. Refactor it
    shape: tuple[int, int, int] = (input_shape.width, input_shape.height, 3)
    inp = Input(shape=shape)

    x = ZeroPadding2D(padding=(3, 3), input_shape=shape)(inp)
    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name="bn1")(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name="lrn_1")(x)
    x = Conv2D(64, (1, 1), name="conv2")(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name="bn2")(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name="conv3")(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name="bn3")(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name="lrn_2")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name="inception_3a_3x3_conv1")(x)
    inception_3a_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3a_3x3_bn1"
    )(inception_3a_3x3)
    inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name="inception_3a_3x3_conv2")(
        inception_3a_3x3
    )
    inception_3a_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3a_3x3_bn2"
    )(inception_3a_3x3)
    inception_3a_3x3 = Activation("relu")(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name="inception_3a_5x5_conv1")(x)
    inception_3a_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3a_5x5_bn1"
    )(inception_3a_5x5)
    inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name="inception_3a_5x5_conv2")(
        inception_3a_5x5
    )
    inception_3a_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3a_5x5_bn2"
    )(inception_3a_5x5)
    inception_3a_5x5 = Activation("relu")(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name="inception_3a_pool_conv")(
        inception_3a_pool
    )
    inception_3a_pool = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3a_pool_bn"
    )(inception_3a_pool)
    inception_3a_pool = Activation("relu")(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name="inception_3a_1x1_conv")(x)
    inception_3a_1x1 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3a_1x1_bn"
    )(inception_3a_1x1)
    inception_3a_1x1 = Activation("relu")(inception_3a_1x1)

    inception_3a = concatenate(
        [inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1],
        axis=3,
    )

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name="inception_3b_3x3_conv1")(inception_3a)
    inception_3b_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3b_3x3_bn1"
    )(inception_3b_3x3)
    inception_3b_3x3 = Activation("relu")(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name="inception_3b_3x3_conv2")(
        inception_3b_3x3
    )
    inception_3b_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3b_3x3_bn2"
    )(inception_3b_3x3)
    inception_3b_3x3 = Activation("relu")(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name="inception_3b_5x5_conv1")(inception_3a)
    inception_3b_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3b_5x5_bn1"
    )(inception_3b_5x5)
    inception_3b_5x5 = Activation("relu")(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name="inception_3b_5x5_conv2")(
        inception_3b_5x5
    )
    inception_3b_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3b_5x5_bn2"
    )(inception_3b_5x5)
    inception_3b_5x5 = Activation("relu")(inception_3b_5x5)

    inception_3b_pool = Lambda(lambda x: x**2, name="power2_3b")(inception_3a)
    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(
        inception_3b_pool
    )
    inception_3b_pool = Lambda(lambda x: x * 9, name="mult9_3b")(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: K.sqrt(x), name="sqrt_3b")(inception_3b_pool)
    inception_3b_pool = Conv2D(64, (1, 1), name="inception_3b_pool_conv")(
        inception_3b_pool
    )
    inception_3b_pool = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3b_pool_bn"
    )(inception_3b_pool)
    inception_3b_pool = Activation("relu")(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name="inception_3b_1x1_conv")(inception_3a)
    inception_3b_1x1 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3b_1x1_bn"
    )(inception_3b_1x1)
    inception_3b_1x1 = Activation("relu")(inception_3b_1x1)

    inception_3b = concatenate(
        [inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1],
        axis=3,
    )

    # Inception3c
    inception_3c_3x3 = Conv2D(
        128, (1, 1), strides=(1, 1), name="inception_3c_3x3_conv1"
    )(inception_3b)
    inception_3c_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3c_3x3_bn1"
    )(inception_3c_3x3)
    inception_3c_3x3 = Activation("relu")(inception_3c_3x3)
    inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
    inception_3c_3x3 = Conv2D(
        256, (3, 3), strides=(2, 2), name="inception_3c_3x3_conv" + "2"
    )(inception_3c_3x3)
    inception_3c_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3c_3x3_bn" + "2"
    )(inception_3c_3x3)
    inception_3c_3x3 = Activation("relu")(inception_3c_3x3)

    inception_3c_5x5 = Conv2D(
        32, (1, 1), strides=(1, 1), name="inception_3c_5x5_conv1"
    )(inception_3b)
    inception_3c_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3c_5x5_bn1"
    )(inception_3c_5x5)
    inception_3c_5x5 = Activation("relu")(inception_3c_5x5)
    inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
    inception_3c_5x5 = Conv2D(
        64, (5, 5), strides=(2, 2), name="inception_3c_5x5_conv" + "2"
    )(inception_3c_5x5)
    inception_3c_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_3c_5x5_bn" + "2"
    )(inception_3c_5x5)
    inception_3c_5x5 = Activation("relu")(inception_3c_5x5)

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate(
        [inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3
    )

    # inception 4a
    inception_4a_3x3 = Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_4a_3x3_conv" + "1"
    )(inception_3c)
    inception_4a_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "1"
    )(inception_4a_3x3)
    inception_4a_3x3 = Activation("relu")(inception_4a_3x3)
    inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
    inception_4a_3x3 = Conv2D(
        192, (3, 3), strides=(1, 1), name="inception_4a_3x3_conv" + "2"
    )(inception_4a_3x3)
    inception_4a_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "2"
    )(inception_4a_3x3)
    inception_4a_3x3 = Activation("relu")(inception_4a_3x3)

    inception_4a_5x5 = Conv2D(
        32, (1, 1), strides=(1, 1), name="inception_4a_5x5_conv1"
    )(inception_3c)
    inception_4a_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4a_5x5_bn1"
    )(inception_4a_5x5)
    inception_4a_5x5 = Activation("relu")(inception_4a_5x5)
    inception_4a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5)
    inception_4a_5x5 = Conv2D(
        64, (5, 5), strides=(1, 1), name="inception_4a_5x5_conv" + "2"
    )(inception_4a_5x5)
    inception_4a_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4a_5x5_bn" + "2"
    )(inception_4a_5x5)
    inception_4a_5x5 = Activation("relu")(inception_4a_5x5)

    inception_4a_pool = Lambda(lambda x: x**2, name="power2_4a")(inception_3c)
    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(
        inception_4a_pool
    )
    inception_4a_pool = Lambda(lambda x: x * 9, name="mult9_4a")(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: K.sqrt(x), name="sqrt_4a")(inception_4a_pool)

    inception_4a_pool = Conv2D(
        128, (1, 1), strides=(1, 1), name="inception_4a_pool_conv" + ""
    )(inception_4a_pool)
    inception_4a_pool = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4a_pool_bn" + ""
    )(inception_4a_pool)
    inception_4a_pool = Activation("relu")(inception_4a_pool)
    inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

    inception_4a_1x1 = Conv2D(
        256, (1, 1), strides=(1, 1), name="inception_4a_1x1_conv" + ""
    )(inception_3c)
    inception_4a_1x1 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4a_1x1_bn" + ""
    )(inception_4a_1x1)
    inception_4a_1x1 = Activation("relu")(inception_4a_1x1)

    inception_4a = concatenate(
        [inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1],
        axis=3,
    )

    # inception4e
    inception_4e_3x3 = Conv2D(
        160, (1, 1), strides=(1, 1), name="inception_4e_3x3_conv" + "1"
    )(inception_4a)
    inception_4e_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4e_3x3_bn" + "1"
    )(inception_4e_3x3)
    inception_4e_3x3 = Activation("relu")(inception_4e_3x3)
    inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
    inception_4e_3x3 = Conv2D(
        256, (3, 3), strides=(2, 2), name="inception_4e_3x3_conv" + "2"
    )(inception_4e_3x3)
    inception_4e_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4e_3x3_bn" + "2"
    )(inception_4e_3x3)
    inception_4e_3x3 = Activation("relu")(inception_4e_3x3)

    inception_4e_5x5 = Conv2D(
        64, (1, 1), strides=(1, 1), name="inception_4e_5x5_conv" + "1"
    )(inception_4a)
    inception_4e_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "1"
    )(inception_4e_5x5)
    inception_4e_5x5 = Activation("relu")(inception_4e_5x5)
    inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
    inception_4e_5x5 = Conv2D(
        128, (5, 5), strides=(2, 2), name="inception_4e_5x5_conv" + "2"
    )(inception_4e_5x5)
    inception_4e_5x5 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "2"
    )(inception_4e_5x5)
    inception_4e_5x5 = Activation("relu")(inception_4e_5x5)

    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate(
        [inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3
    )

    # inception5a
    inception_5a_3x3 = Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5a_3x3_conv" + "1"
    )(inception_4e)
    inception_5a_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5a_3x3_bn" + "1"
    )(inception_5a_3x3)
    inception_5a_3x3 = Activation("relu")(inception_5a_3x3)
    inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
    inception_5a_3x3 = Conv2D(
        384, (3, 3), strides=(1, 1), name="inception_5a_3x3_conv" + "2"
    )(inception_5a_3x3)
    inception_5a_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5a_3x3_bn" + "2"
    )(inception_5a_3x3)
    inception_5a_3x3 = Activation("relu")(inception_5a_3x3)

    inception_5a_pool = Lambda(lambda x: x**2, name="power2_5a")(inception_4e)
    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(
        inception_5a_pool
    )
    inception_5a_pool = Lambda(lambda x: x * 9, name="mult9_5a")(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: K.sqrt(x), name="sqrt_5a")(inception_5a_pool)

    inception_5a_pool = Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5a_pool_conv" + ""
    )(inception_5a_pool)
    inception_5a_pool = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5a_pool_bn" + ""
    )(inception_5a_pool)
    inception_5a_pool = Activation("relu")(inception_5a_pool)
    inception_5a_pool = ZeroPadding2D(padding=(1, 1))(inception_5a_pool)

    inception_5a_1x1 = Conv2D(
        256, (1, 1), strides=(1, 1), name="inception_5a_1x1_conv" + ""
    )(inception_4e)
    inception_5a_1x1 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5a_1x1_bn" + ""
    )(inception_5a_1x1)
    inception_5a_1x1 = Activation("relu")(inception_5a_1x1)

    inception_5a = concatenate(
        [inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3
    )

    # inception_5b
    inception_5b_3x3 = Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5b_3x3_conv" + "1"
    )(inception_5a)
    inception_5b_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5b_3x3_bn" + "1"
    )(inception_5b_3x3)
    inception_5b_3x3 = Activation("relu")(inception_5b_3x3)
    inception_5b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3)
    inception_5b_3x3 = Conv2D(
        384, (3, 3), strides=(1, 1), name="inception_5b_3x3_conv" + "2"
    )(inception_5b_3x3)
    inception_5b_3x3 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5b_3x3_bn" + "2"
    )(inception_5b_3x3)
    inception_5b_3x3 = Activation("relu")(inception_5b_3x3)

    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)

    inception_5b_pool = Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5b_pool_conv" + ""
    )(inception_5b_pool)
    inception_5b_pool = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5b_pool_bn" + ""
    )(inception_5b_pool)
    inception_5b_pool = Activation("relu")(inception_5b_pool)

    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = Conv2D(
        256, (1, 1), strides=(1, 1), name="inception_5b_1x1_conv" + ""
    )(inception_5a)
    inception_5b_1x1 = BatchNormalization(
        axis=3, epsilon=0.00001, name="inception_5b_1x1_bn" + ""
    )(inception_5b_1x1)
    inception_5b_1x1 = Activation("relu")(inception_5b_1x1)

    inception_5b = concatenate(
        [inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3
    )

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name="dense_layer")(reshape_layer)
    norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(
        dense_layer
    )

    # Final Model
    return Model(inputs=[inp], outputs=norm_layer)
