import os
import gdown
import numpy
import cv2
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.models.analyzer import Analyzer

logger = Logger(module="extendedmodels.Emotion")

# -------------------------------------------
# pylint: disable=line-too-long
# -------------------------------------------
# dependency configuration
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )
# -------------------------------------------

# Labels for the emotions that can be detected by the model.
labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


# pylint: disable=too-few-public-methods
class EmotionClient(Analyzer):

    _model: Sequential  # The actual model used for the analysis

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def predict(self, img: numpy.ndarray) -> numpy.ndarray:
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = numpy.expand_dims(img_gray, axis=0)

        emotion_predictions = self.model.predict(img_gray, verbose=0)[0, :]
        return emotion_predictions

    def __initialize(self) -> Sequential:

        num_classes = 7

        self._model = Sequential()

        # 1st convolution layer
        self._model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
        self._model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        self._model.add(Conv2D(64, (3, 3), activation="relu"))
        self._model.add(Conv2D(64, (3, 3), activation="relu"))
        self._model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        self._model.add(Conv2D(128, (3, 3), activation="relu"))
        self._model.add(Conv2D(128, (3, 3), activation="relu"))
        self._model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self._model.add(Flatten())

        # fully connected neural networks
        self._model.add(Dense(1024, activation="relu"))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(1024, activation="relu"))
        self._model.add(Dropout(0.2))

        self._model.add(Dense(num_classes, activation="softmax"))

        file_name = "facial_expression_model_weights.h5"
        url = (
            f"https://github.com/serengil/deepface_models/releases/download/v1.0/{file_name}",
        )
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")
            gdown.download(url, output, quiet=False)

        self._model.load_weights(output)
