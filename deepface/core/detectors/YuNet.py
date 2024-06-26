from typing import Any, Dict, List, Optional

import os
import cv2
import numpy
import gdown

from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase
from deepface.commons.logger import Logger
from deepface.core.exceptions import (
    FaceNotFoundError,
    MissingDependencyError,
    InsufficentVersionError,
)
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)

opencv_version = cv2.__version__.split(".")
if not len(opencv_version) >= 2:
    raise InsufficentVersionError(
        f"{__name__} requires opencv-python >= 4.8 but you have {cv2.__version__}"
    )

opencv_version_major = int(opencv_version[0])
opencv_version_minor = int(opencv_version[1])
if opencv_version_major < 4 or (opencv_version_major == 4 and opencv_version_minor < 8):
    raise InsufficentVersionError(
        f"{__name__} requires opencv-python >= 4.8 but you have {cv2.__version__}"
    )

try:
    # pylint: disable=unused-import
    from cv2 import dnn

    # pylint: enable=unused-import
except ModuleNotFoundError:
    what: str = f"{__name__} requires `opencv-contrib-python` library."
    what += "You can install by 'pip install opencv-contrib-python' "
    raise MissingDependencyError(what) from None

logger = Logger.get_instance()


# YuNet detector (optional)
class Detector(DetectorBase):

    _detector: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._KDEFAULT_MIN_CONFIDENCE = float(0.8)
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
        tag: Optional[str] = None,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: Optional[float] = None,
        key_points: bool = True,
        raise_notfound: bool = False,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, tag, min_dims, min_confidence, key_points, raise_notfound)

        if min_dims is None:
            min_dims = BoxDimensions(width=0, height=0)
        if min_confidence is None:
            min_confidence = self._KDEFAULT_MIN_CONFIDENCE
        if min_confidence < 0 or min_confidence > 1:
            raise ValueError(
                f"min_confidence must be in the range [0, 1]. Got {min_confidence}."
            )

        detected_faces: List[DetectedFace] = []
        img_height, img_width, *_ = img.shape
        processed_img = img

        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        scale_factor = min(640.0 / max(img_height, img_width), 1.0)
        if scale_factor < 1.0:
            img_height, img_width = tuple(
                map(int, (img_height * scale_factor, img_width * scale_factor))
            )
            processed_img = cv2.resize(img, (img_width, img_height))
        else:
            scale_factor = 1.0

        self._detector.setInputSize((img_width, img_height))
        self._detector.setScoreThreshold(min_confidence)
        _, faces = self._detector.detect(processed_img)
        if faces is not None:
            for face in faces:

                confidence = round(float(face[-1]), 5)
                if confidence < min_confidence:
                    continue  # Skip low confidence detections

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
                (
                    x,
                    y,
                    w,
                    h,
                    x_re,
                    y_re,
                    x_le,
                    y_le,
                    x_nt,
                    y_nt,
                    x_rcm,
                    y_rcm,
                    x_lcm,
                    y_lcm,
                ) = [int(round(value / scale_factor)) for value in face[:14]]
                x_range = RangeInt(x, min(x + w, img_width))
                y_range = RangeInt(y, min(y + h, img_height))
                if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                    continue  # Invalid or empty detection

                bounding_box: BoundingBox = BoundingBox(
                    top_left=Point(x=x, y=y),
                    bottom_right=Point(x=x_range.end, y=y_range.end),
                )

                points: Optional[Dict[str, Point]] = None
                if key_points:
                    le_point = Point(x=x_le, y=y_le)
                    re_point = Point(x=x_re, y=y_re)
                    nt_point = Point(x=x_nt, y=y_nt)
                    lm_point = Point(x=x_lcm, y=y_lcm)
                    rm_point = Point(x=x_rcm, y=y_rcm)
                    mc_point = Point(
                        x=(x_lcm + x_rcm) // 2, y=(y_lcm + y_rcm) // 2
                    )
                    points = {
                        "lec": le_point,
                        "rec": re_point,
                        "nt": nt_point,
                        "mlc": lm_point,
                        "mrc": rm_point,
                        "mc": mc_point,
                    }

                    # Remove any points that are outside the bounding box
                    for key in list(points.keys()):
                        pt: Optional[Point] = points[key]
                        if pt is not None and pt not in bounding_box:
                            points.pop(key)

                try:
                    # This might raise an exception if the values are out of bounds
                    detected_faces.append(
                        DetectedFace(
                            confidence=confidence,
                            bounding_box=bounding_box,
                            key_points=points,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error: {e}")

        if 0 == len(detected_faces) and raise_notfound:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            img=img,
            tag=tag,
            detections=detected_faces,
        )
