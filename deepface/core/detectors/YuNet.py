from typing import Any, List, Optional

import os
import cv2
import numpy
import gdown

from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase
from deepface.commons.logger import Logger
from deepface.core.exceptions import (
    FaceNotFound,
    MissingOptionalDependency,
    InsufficentVersionRequirement,
)
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace, Point, RangeInt

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

# YuNet detector (optional)
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

    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.9,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        img_height, img_width = img.shape[:2]
        processed_img = img
        detected_faces: List[DetectedFace] = []

        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        scale_factor = min(640.0 / max(img_height, img_width), 1.0)
        if scale_factor < 1.0:
            img_height, img_width = tuple(
                map(int, (img_height * scale_factor, img_width * scale_factor))
            )
            processed_img = cv2.resize(img, (img_width, img_height))

        self._detector.setInputSize((img_width, img_height))
        self._detector.setScoreThreshold(min_confidence)
        _, faces = self._detector.detect(processed_img)
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
            (x, y, w, h, x_re, y_re, x_le, y_le) = [int(coord / scale_factor) for coord in face[:8]]
            x_range = RangeInt(x, min(x + w, img_width))
            y_range = RangeInt(y, min(y + h, img_height))
            if x_range.span <= 0 or y_range.span <= 0:
                continue
            if isinstance(min_dims, BoxDimensions):
                if min_dims.width > 0 and x_range.span < min_dims.width:
                    continue
                if min_dims.height > 0 and y_range.span < min_dims.height:
                    continue

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x, y=y),
                bottom_right=Point(x=x + w, y=y + h),
            )

            le_point = Point(x=x_le, y=y_le)
            re_point = Point(x=x_re, y=y_re)
            if le_point not in bounding_box or re_point not in bounding_box:
                le_point = None
                re_point = None

            confidence = float(face[-1])
            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=confidence,
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFound("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
