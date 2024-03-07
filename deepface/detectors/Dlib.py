from typing import Any, List
import os
import bz2
import gdown
import numpy
from deepface.commons import folder_utils
from deepface.models.Detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger(module="detectors.DlibWrapper")


class Detector(DetectorBase):
    """
    This class is used to detect faces using dlib's hog face detector.
    Note! This is an optional detector, ensure the library is installed.
    """

    _detector: Any
    _predictor: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1))
        self.__initialize()

    def __initialize(self):
        try:
            import dlib  # Lazy import.

            file_name = "shape_predictor_5_face_landmarks.dat"
            weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

            # check required file exists in the home/.deepface/weights folder
            if os.path.isfile(weight_file) != True:

                logger.info(f"Download : {file_name}")
                source_file = f"{file_name}.bz2"

                url = f"http://dlib.net/files/{source_file}"
                dest = os.path.join(folder_utils.get_weights_dir(), source_file)
                gdown.download(url, dest, quiet=False)

                zipfile = bz2.BZ2File(dest)
                data = zipfile.read()
                with open(weight_file, "wb") as f:
                    f.write(data)

            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(weight_file)

        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional detector, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with dlib

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
        detections, scores, _ = self._detector.run(img, 1)

        if len(detections) > 0:

            for idx, detection in enumerate(detections):
                left = detection.left()
                right = detection.right()
                top = detection.top()
                bottom = detection.bottom()

                y = int(max(0, top))
                h = int(min(bottom, img.shape[0]) - y)
                x = int(max(0, left))
                w = int(min(right, img.shape[1]) - x)

                shape = self._predictor(img, detection)
                left_eye = (shape.part(2).x, shape.part(2).y)
                right_eye = (shape.part(0).x, shape.part(0).y)

                # never saw confidence higher than +3.5 github.com/davisking/dlib/issues/761
                confidence = scores[idx]

                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=min(max(0, confidence), 1.0),
                )
                results.append(facial_area)

        return results
