import os
from typing import List

import numpy
import gdown

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import MissingOptionalDependency
from deepface.core.extractor import Extractor as ExtractorBase
from deepface.core.types import BoxDimensions

try:
    from cv2 import FaceRecognizerSF
except ModuleNotFoundError:
    what: str = f"{__name__} requires `opencv-contrib-python` library."
    what += "You can install by 'pip install opencv-contrib-python' "
    raise MissingOptionalDependency(what) from None


logger = Logger.get_instance()


# pylint: disable=too-few-public-methods
class Extractor(ExtractorBase):

    _model: FaceRecognizerSF

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=112, height=112)
        self._output_shape = 128
        self._initialize()

    def _initialize(self):
        weights_folder = folder_utils.get_weights_dir()
        file_name = "face_recognition_sface_2021dec.onnx"
        output = os.path.join(weights_folder, file_name)

        if not os.path.isfile(output):
            logger.info(f"Download : {file_name}")
            url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/"
            url += f"face_recognition_sface/{file_name}"
            gdown.download(url, output, quiet=False)

        self._model = FaceRecognizerSF.create(
            model=output, config="", backend_id=0, target_id=0
        )

    def process(self, img: numpy.ndarray) -> List[float]:

        super().process(img)
        # TODO: shouldn't we ensure image is resized to fit in the input_shape?

        # revert the image to original format and preprocess using the model
        input_blob = (img[0] * 255).astype(numpy.uint8)
        embeddings = self._model.feature(input_blob)
        return embeddings[0].tolist()

