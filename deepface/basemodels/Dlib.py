from typing import List
import os
import bz2
import gdown
import numpy
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.decomposer import Decomposer

logger = Logger.get_instance()

class DlibClient(Decomposer):
    """
    Dlib model class
    """

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.model = DlibResNet()
        self.input_shape = (150, 150)
        self.output_shape = 128

    def process(self, img: numpy.ndarray) -> List[float]:
        """
        find embeddings with Dlib model - different than regular models
        Args:
            img (numpy.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # return self.model.predict(img)[0].tolist()

        # detect_faces returns 4 dimensional images
        if len(img.shape) == 4:
            img = img[0]

        # bgr to rgb
        img = img[:, :, ::-1]  # bgr to rgb

        # img is in scale of [0, 1] but expected [0, 255]
        if img.max() <= 1:
            img = img * 255

        img = img.astype(numpy.uint8)

        img_representation = self.model.model.compute_face_descriptor(img)
        img_representation = numpy.array(img_representation)
        img_representation = numpy.expand_dims(img_representation, axis=0)
        return img_representation[0].tolist()


class DlibResNet:
    def __init__(self):

        ## this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional dependency, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        self.layers = [DlibMetaData()]

        # ---------------------

        file_name: str = "dlib_face_recognition_resnet_model_v1.dat"
        url: str = f"http://dlib.net/files/{file_name}.bz2"
        output: str = os.path.join(folder_utils.get_weights_dir(), file_name)

        # ---------------------

        # download pre-trained model if it does not exist
        if os.path.isfile(output) != True:
            logger.info(f"Download : {file_name}")

            compressed_output = output + ".bz2"
            gdown.download(url, compressed_output, quiet=False)

            zipfile = bz2.BZ2File(compressed_output)
            data = zipfile.read()
            with open(output, "wb") as f:
                f.write(data)

            # remove the downloaded file
            os.remove(output)

        # ---------------------

        self.model = dlib.face_recognition_model_v1(output)

        # ---------------------


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
