from typing import Dict, Union

import os
import gdown
import numpy
import cv2

from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.core.analyzer import Analyzer as AnalyzerBase

logger = Logger.get_instance()

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


# pylint: disable=too-few-public-methods
class Analyzer(AnalyzerBase):

    _model: Sequential  # The actual model used for the analysis
    _labels = [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
    ]

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def process(
        self, img: numpy.ndarray, detail: bool = False
    ) -> Dict[str, Union[str, Dict[str, float]]]:

        result = {}
        attribute = self.name.lower()

        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = numpy.expand_dims(img_gray, axis=0)

        estimates = self._model.predict(img_gray, verbose=0)[0, :]
        result[attribute] = self._labels[numpy.argmax(estimates)]

        if detail:
            details = {}
            estimates_sum = numpy.sum(estimates)
            for i, label in enumerate(self._labels):
                estimate = round(estimates[i] * 100 / estimates_sum, 2)
                details[label] = estimate
            result[f"{attribute}_analysis"] = details

        return result

    def __initialize(self) -> Sequential:

        file_name = "facial_expression_model_weights.h5"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            url = "https://github.com/serengil/deepface_models/releases/"
            url += f"download/v1.0/{file_name}"
            gdown.download(url, weight_file, quiet=False)

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
        self._model.load_weights(weight_file)
