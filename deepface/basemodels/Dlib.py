from typing import List
import os
import bz2
import gdown
import numpy
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger(module="basemodels.DlibResNet")

class DlibClient(FacialRecognition):
    """
    Dlib model class
    """

    def __init__(self):
        self.model = DlibResNet()
        self.model_name = "Dlib"
        self.input_shape = (150, 150)
        self.output_shape = 128

    def find_embeddings(self, img: numpy.ndarray) -> List[float]:
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

        home = folder_utils.get_deepface_home()
        weight_file = home + "/.deepface/weights/dlib_face_recognition_resnet_model_v1.dat"

        # ---------------------

        # download pre-trained model if it does not exist
        if os.path.isfile(weight_file) != True:
            logger.info("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")

            file_name = "dlib_face_recognition_resnet_model_v1.dat.bz2"
            url = f"http://dlib.net/files/{file_name}"
            output = f"{home}/.deepface/weights/{file_name}"
            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            newfilepath = output[:-4]  # discard .bz2 extension
            with open(newfilepath, "wb") as f:
                f.write(data)

        # ---------------------

        self.model = dlib.face_recognition_model_v1(weight_file)

        # ---------------------

class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
