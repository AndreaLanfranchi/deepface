from typing import Any, List

import os
import cv2
import numpy
import gdown

from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons.logger import Logger
from deepface.core.exceptions import (
    MissingOptionalDependency,
    InsufficentVersionRequirement,
)

opencv_version = cv2.__version__.split(".")
if not len(opencv_version) >= 2:
    raise InsufficentVersionRequirement(
        f"{__name__} requires opencv-python >= 4.8 but you have {cv2.__version__}"
    )

opencv_version_major = int(opencv_version[0])
opencv_version_minor = int(opencv_version[1])
if opencv_version_major < 4 or (opencv_version_major == 4 and opencv_version_minor < 8):
    raise InsufficentVersionRequirement(
        f"{__name__} requires opencv-python >= 4.8 but you have {cv2.__version__}"
    )

try:
    # pylint: disable=unused-import
    from cv2 import dnn

    # pylint: enable=unused-import
except ModuleNotFoundError:
    what: str = f"{__name__} requires `opencv-contrib-python` library."
    what += "You can install by 'pip install opencv-contrib-python' "
    raise MissingOptionalDependency(what) from None

logger = Logger.get_instance()

# YuNet detector
class Detector(DetectorBase):

    _detector: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self) -> Any:

        file_name = "face_detection_yunet_2023mar.onnx"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

        if os.path.isfile(weight_file) is False:
            logger.info(f"Download : {file_name}")

            url = "https://github.com/opencv/opencv_zoo/raw/main/models/"
            url += f"face_detection_yunet/{file_name}"
            gdown.download(url, weight_file, quiet=False)

        self._detector = cv2.FaceDetectorYN_create(weight_file, "", (0, 0))

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:

        # TODO: remove ! FaceDetector.detect_faces does not support score_threshold parameter.
        # We can set it via environment variable.
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))

        results = []

        height, width = img.shape[0], img.shape[1]
        if height == 0 or width == 0:
            return results

        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        scale_factor = 640.0 / max(height, width)
        if scale_factor < 1.0:
            img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
            height, width = img.shape[0], img.shape[1]

        self._detector.setInputSize((width, height))
        self._detector.setScoreThreshold(score_threshold)
        _, faces = self._detector.detect(img)
        if faces is None:
            return results

        for face in faces:

            # The detection output faces is a two-dimension array of type CV_32F,
            # whose rows are the detected face instances, columns are the location
            # of a face and 5 facial landmarks.
            # The format of each row is as follows:
            # x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            # x_rcm, y_rcm, x_lcm, y_lcm,
            # where x1, y1, w, h are the top-left coordinates, width and height of
            # the face bounding box,
            # {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye,
            # left eye, nose tip, the right corner and left corner of the mouth respectively.

            (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))

            # Yunet returns negative coordinates if it thinks part of
            # the detected face is outside the frame.
            # We set the coordinate to 0 if they are negative.
            x = max(x, 0)
            y = max(y, 0)
            if scale_factor < 1.0:
                x, y, w, h = (
                    int(x / scale_factor),
                    int(y / scale_factor),
                    int(w / scale_factor),
                    int(h / scale_factor),
                )
                x_re, y_re, x_le, y_le = (
                    int(x_re / scale_factor),
                    int(y_re / scale_factor),
                    int(x_le / scale_factor),
                    int(y_le / scale_factor),
                )

            left_eye = (x_re, y_re)
            right_eye = (x_le, y_le)
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
