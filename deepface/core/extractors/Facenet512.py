from typing import List, Optional, Union

import os
import tensorflow
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import InsufficentVersionError
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace
from deepface.core.extractor import Extractor as ExtractorBase
from deepface.core.extractors.Facenet128 import InceptionResNetV1

tensorflow_version_major = int(tensorflow.__version__.split(".", maxsplit=1)[0])
if tensorflow_version_major < 2:
    raise InsufficentVersionError("Tensorflow reequires version >=2.0.0")

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
from tensorflow.keras.models import Model

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
