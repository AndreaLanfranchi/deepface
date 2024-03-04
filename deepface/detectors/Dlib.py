from typing import List
import os
import bz2
import gdown
import numpy
from deepface.commons import folder_utils
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger(module="detectors.DlibWrapper")


class DlibClient(Detector):

    def __init__(self):
        self.name = "dlib"
        self.model = self.build_model()

    def build_model(self) -> dict:
        """
        Build a dlib hog face detector model
        Returns:
            model (Any)
        """

        # this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional detector, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        file_name = "shape_predictor_5_face_landmarks.dat"
        output = os.path.join(folder_utils.get_weights_dir(), file_name)

        # check required file exists in the home/.deepface/weights folder
        if os.path.isfile(output) != True:

            logger.info(f"Download : {file_name}")
            source_file = f"{file_name}.bz2"

            url = f"http://dlib.net/files/{source_file}"
            dest = os.path.join(folder_utils.get_weights_dir(), source_file)
            gdown.download(url, dest, quiet=False)

            zipfile = bz2.BZ2File(dest)
            data = zipfile.read()
            with open(output, "wb") as f:
                f.write(data)

        face_detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(output)

        detector = {}
        detector["face_detector"] = face_detector
        detector["sp"] = sp
        return detector

    def detect_faces(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with dlib

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        face_detector = self.model["face_detector"]

        # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
        detections, scores, _ = face_detector.run(img, 1)

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

                shape = self.model["sp"](img, detection)
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
