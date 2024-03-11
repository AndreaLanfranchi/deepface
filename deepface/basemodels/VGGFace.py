from typing import List
import os
import gdown
import numpy
from deepface.commons import package_utils, folder_utils
from deepface.core.types import BoxDimensions
from deepface.modules import verification
from deepface.core.decomposer import Representer
from deepface.commons.logger import Logger

logger = Logger.get_instance()

# ---------------------------------------

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )

# ---------------------------------------


# pylint: disable=too-few-public-methods
class VggFaceClient(Representer):
    """
    VGG-Face model class
    """

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(224, 224)
        self._output_shape = 4096
        self._initialize()

    def process(self, img: numpy.ndarray) -> List[float]:
        """
        find embeddings with VGG-Face model
        Args:
            img (numpy.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        # having normalization layer in descriptor troubles for some gpu users (e.g. issue 957, 966)
        # instead we are now calculating it with traditional way not with keras backend
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
        ret = Sequential()
        ret.add(
            ZeroPadding2D(
                (1, 1), input_shape=(self.input_shape.width, self.input_shape.height, 3)
            )
        )
        ret.add(Convolution2D(64, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(64, (3, 3), activation="relu"))
        ret.add(MaxPooling2D((2, 2), strides=(2, 2)))

        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(128, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(128, (3, 3), activation="relu"))
        ret.add(MaxPooling2D((2, 2), strides=(2, 2)))

        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(256, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(256, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(256, (3, 3), activation="relu"))
        ret.add(MaxPooling2D((2, 2), strides=(2, 2)))

        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(512, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(512, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(512, (3, 3), activation="relu"))
        ret.add(MaxPooling2D((2, 2), strides=(2, 2)))

        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(512, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(512, (3, 3), activation="relu"))
        ret.add(ZeroPadding2D((1, 1)))
        ret.add(Convolution2D(512, (3, 3), activation="relu"))
        ret.add(MaxPooling2D((2, 2), strides=(2, 2)))

        ret.add(Convolution2D(self._output_shape, (7, 7), activation="relu"))
        ret.add(Dropout(0.5))
        ret.add(Convolution2D(self._output_shape, (1, 1), activation="relu"))
        ret.add(Dropout(0.5))
        ret.add(Convolution2D(2622, (1, 1)))
        ret.add(Flatten())
        ret.add(Activation("softmax"))

        return ret
