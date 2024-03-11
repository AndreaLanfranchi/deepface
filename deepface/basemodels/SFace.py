import os
from typing import Any, List

import numpy
import cv2
import gdown

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.decomposer import Representer

logger = Logger.get_instance()


# pylint: disable=too-few-public-methods
class SFaceClient(Representer):
    """
    SFace model class
    """

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.model = load_model()
        self.input_shape = (112, 112)
        self.output_shape = 128

    def process(self, img: numpy.ndarray) -> List[float]:
        """
        find embeddings with SFace model - different than regular models
        Args:
            img (numpy.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # return self.model.predict(img)[0].tolist()

        # revert the image to original format and preprocess using the model
        input_blob = (img[0] * 255).astype(numpy.uint8)

        embeddings = self.model.model.feature(input_blob)

        return embeddings[0].tolist()


def load_model(
    # pylint: disable=line-too-long
    url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    # pylint: enable=line-too-long
) -> Any:
    """
    Construct SFace model, download its weights and load
    """

    file_name = "face_recognition_sface_2021dec.onnx"
    output = os.path.join(folder_utils.get_weights_dir(), file_name)

    if not os.path.isfile(output):
        logger.info(f"Download : {file_name}")
        gdown.download(url, output, quiet=False)

    model = SFaceWrapper(model_path=output)
    return model


class SFaceWrapper:
    def __init__(self, model_path):
        """
        SFace wrapper covering model construction, layer infos and predict
        """
        try:
            self.model = cv2.FaceRecognizerSF.create(
                model=model_path, config="", backend_id=0, target_id=0
            )
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv2.FaceRecognizerSF module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

        self.layers = [_Layer()]


class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)
