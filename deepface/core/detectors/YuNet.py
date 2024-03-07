from typing import Any, List

import os
import cv2
import numpy
import gdown

from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger(module="detectors.YunetWrapper")


class Detector(DetectorBase):

    _detector: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1))
        self.__initialize()

    def __initialize(self) -> Any:

        try:
            opencv_version = cv2.__version__.split(".")
            if (
                len(opencv_version) > 2
                and int(opencv_version[0]) == 4
                and int(opencv_version[1]) < 8
            ):
                # min requirement: https://github.com/opencv/opencv_zoo/issues/172
                raise RuntimeError(
                    f"YuNet requires opencv-python >= 4.8 but you have {cv2.__version__}"
                )

            # pylint: disable=C0301
            file_name = "face_detection_yunet_2023mar.onnx"
            url = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{file_name}"
            output = os.path.join(folder_utils.get_weights_dir(), file_name)

            if os.path.isfile(output) is False:
                logger.info(f"Download : {file_name}")
                gdown.download(url, output, quiet=False)

            self._detector = cv2.FaceDetectorYN_create(output, "", (0, 0))

        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.FaceDetectorYN_create module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with yunet

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        # FaceDetector.detect_faces does not support score_threshold parameter.
        # We can set it via environment variable.
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))

        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        faces = []
        height, width = img.shape[0], img.shape[1]
        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        resized = False
        if height > 640 or width > 640:
            r = 640.0 / max(height, width)
            original_image = img.copy()
            img = cv2.resize(img, (int(width * r), int(height * r)))
            height, width = img.shape[0], img.shape[1]
            resized = True
        self._detector.setInputSize((width, height))
        self._detector.setScoreThreshold(score_threshold)
        _, faces = self._detector.detect(img)
        if faces is None:
            return results
        for face in faces:
            # pylint: disable=W0105
            """
            The detection output faces is a two-dimension array of type CV_32F,
            whose rows are the detected face instances, columns are the location
            of a face and 5 facial landmarks.
            The format of each row is as follows:
            x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            x_rcm, y_rcm, x_lcm, y_lcm,
            where x1, y1, w, h are the top-left coordinates, width and height of
            the face bounding box,
            {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye,
            left eye, nose tip, the right corner and left corner of the mouth respectively.
            """
            (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))
            left_eye = (x_re, y_re)
            right_eye = (x_le, y_le)

            # Yunet returns negative coordinates if it thinks part of
            # the detected face is outside the frame.
            # We set the coordinate to 0 if they are negative.
            x = max(x, 0)
            y = max(y, 0)
            if resized:
                img = original_image
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                )
            confidence = float(face[-1])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
            )
            results.append(facial_area)
        return results
