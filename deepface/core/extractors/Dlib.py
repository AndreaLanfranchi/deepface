from typing import List, Optional, Union

import os
import bz2
import shutil
import cv2
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.extractor import Extractor as ExtractorBase
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace
from deepface.core.exceptions import MissingDependencyError

try:
    import dlib
except ModuleNotFoundError:
    what: str = "`Dlib` is an optional dependency, ensure the library is installed. "
    what += "You can install by 'pip install dlib' "
    raise MissingDependencyError(what) from None

logger = Logger.get_instance()


# Dlib respresenter model (optional)
class Extractor(ExtractorBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=150, height=150)
        self._output_shape = 128
        self._initialize()

    def process(
        self,
        img: numpy.ndarray,
        face: Optional[Union[DetectedFace, BoundingBox]] = None,
    ) -> List[float]:

        super().process(img, face)
        img = self._to_required_shape(img, face)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_representation = self._model.compute_face_descriptor(img)
        img_representation = numpy.array(img_representation)
        img_representation = numpy.expand_dims(img_representation, axis=0)
        return img_representation[0].tolist()

    def _initialize(self):

        file_name: str = "dlib_face_recognition_resnet_model_v1.dat"
        weight_file: str = os.path.join(folder_utils.get_weights_dir(), file_name)
        if os.path.isfile(weight_file) != True:
            logger.info(f"Download : {file_name}")

            compressed_file_name: str = f"{file_name}.bz2"
            compressed_output = os.path.join(
                folder_utils.get_weights_dir(), compressed_file_name
            )
            url: str = f"http://dlib.net/files/{compressed_file_name}"
            gdown.download(url, compressed_output, quiet=False)
            with bz2.BZ2File(compressed_output, "rb") as fr, open(
                weight_file, "wb"
            ) as fw:
                shutil.copyfileobj(fr, fw)
            os.remove(compressed_output)

        self._model = dlib.face_recognition_model_v1(weight_file)
