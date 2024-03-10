from typing import List

import os
import bz2
import shutil
import gdown
import numpy

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.decomposer import Decomposer
from deepface.core.types import BoxDimensions

logger = Logger.get_instance()

class DlibClient(Decomposer):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=150, height=150)
        self._output_shape = 128
        self._initialize()

    def process(self, img: numpy.ndarray) -> List[float]:

        if len(img.shape) == 4:
            img = img[0]

        # bgr to rgb
        img = img[:, :, ::-1]  # bgr to rgb

        # img is in scale of [0, 1] but expected [0, 255]
        if img.max() <= 1:
            img = img * 255

        img = img.astype(numpy.uint8)

        img_representation = self._model.compute_face_descriptor(img)
        img_representation = numpy.array(img_representation)
        img_representation = numpy.expand_dims(img_representation, axis=0)
        return img_representation[0].tolist()

    def _initialize(self):
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional dependency, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        file_name: str = "dlib_face_recognition_resnet_model_v1.dat"
        output: str = os.path.join(folder_utils.get_weights_dir(), file_name)

        # download pre-trained model if it does not exist
        if os.path.isfile(output) != True:

            logger.info(f"Download : {file_name}")
            
            compressed_file_name: str = f"{file_name}.bz2"
            compressed_output = os.path.join(folder_utils.get_weights_dir(), compressed_file_name)
            url: str = f"http://dlib.net/files/{compressed_file_name}"

            gdown.download(url, compressed_output, quiet=False)
            with bz2.BZ2File(compressed_output, "rb") as fr, open(output, "wb") as fw:
                shutil.copyfileobj(fr, fw)
            os.remove(compressed_output)

        self._model = dlib.face_recognition_model_v1(output)
