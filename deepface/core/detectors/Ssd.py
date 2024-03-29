from typing import List, Optional

import os
import gdown
import cv2
import pandas
import numpy

from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase
from deepface.core.detectors.OpenCv import Detector as OpenCvDetector
from deepface.commons.logger import Logger
from deepface.core.exceptions import MissingDependencyError, FaceNotFoundError
from deepface.core.types import BoundingBox, BoxDimensions, DetectedFace, Point, RangeInt

try:
    from cv2 import dnn as cv2_dnn
    from cv2.typing import MatLike

except ModuleNotFoundError:
    what: str = f"{__name__} requires `opencv-contrib-python` library."
    what += "You can install by 'pip install opencv-contrib-python' "
    raise MissingDependencyError(what) from None

logger = Logger.get_instance()

# OpenCV's Ssd detector (optional)
class Detector(DetectorBase):

    _detector: cv2_dnn.Net
    _opencv_detector: OpenCvDetector

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(width=300, height=300)
        self._initialize()

    def _initialize(self):
        weights_folder = folder_utils.get_weights_dir()
        file_name = "deploy.prototxt"
        output1 = os.path.join(weights_folder, file_name)

        # model structure
        if os.path.isfile(output1) != True:
            logger.info(f"Download : {file_name}")
            # pylint: disable=line-too-long
            url = f"https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/{file_name}"
            # pylint: enable=line-too-long
            gdown.download(url, output1, quiet=False)

        file_name = "res10_300x300_ssd_iter_140000.caffemodel"
        output2 = os.path.join(weights_folder, file_name)

        # pre-trained weights
        if os.path.isfile(output2) != True:
            logger.info(f"Download : {file_name}")
            # pylint: disable=line-too-long
            url = f"https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/{file_name}"
            # pylint: enable=line-too-long
            gdown.download(url, output2, quiet=False)

        self._detector = cv2_dnn.readNetFromCaffe(output1, output2)
        self._opencv_detector = OpenCvDetector()  # Fix: Assign an instance of OpenCvDetector

    def process(
        self,
        img: numpy.ndarray,
        min_dims: Optional[BoxDimensions] = None,
        min_confidence: float = 0.0,
        raise_notfound: bool = False,
        detect_eyes: bool = True,
    ) -> DetectorBase.Results:

        # Validation of inputs
        super().process(img, min_dims, min_confidence)
        img_height, img_width = img.shape[:2]
        detected_faces: List[DetectedFace] = []

        # TODO: resize to a square ?
        aspect_ratio_x = img_width / self._input_shape.width
        aspect_ratio_y = img_height / self._input_shape.height
        target_w, target_h = self._input_shape.width, self._input_shape.height

        blob = cv2_dnn.blobFromImage(
            image=cv2.resize(img, (target_h, target_w)),
            scalefactor=1.0,
            size=(target_h, target_w),
            mean=(104.0, 177.0, 123.0),
        )
        self._detector.setInput(blob)
        detections: MatLike = self._detector.forward()

        detections_df = pandas.DataFrame(
            detections[0][0],
            columns=[
                "img_id",
                "is_face",
                "confidence",
                "left",
                "top",
                "right",
                "bottom",
            ],
        )

        # 0: background, 1: face
        detections_df = detections_df[
            (detections_df["is_face"] == 1) & (detections_df["confidence"] >= 0.90)
        ]
        detections_df[["left", "bottom", "right", "top"]] *= int(300)

        for _, row in detections_df.iterrows():

            confidence = float(row["confidence"])
            if min_confidence is not None and confidence < min_confidence:
                continue

            x1 = int(round(row["left"] * aspect_ratio_x))
            x2 = int(round(row["right"] * aspect_ratio_x))
            y1 = int(round(row["top"] * aspect_ratio_y))
            y2 = int(round(row["bottom"] * aspect_ratio_y))
            x_range = RangeInt(x1, min(x2, img_width))
            y_range = RangeInt(y1, min(y2, img_height))
            if x_range.span <= 0 or y_range.span <= 0:
                continue  # Invalid detection
            if min_dims is not None:
                if min_dims.width > 0 and x_range.span < min_dims.width:
                    continue
                if min_dims.height > 0 and y_range.span < min_dims.height:
                    continue

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            le_point = None
            re_point = None
            if detect_eyes:
                eyes: List[Point] = self._opencv_detector.find_eyes(img[y1:y2, x1:x2])
                if len(eyes) == 2:
                    # Normalize left and right eye coordinates to the whole image
                    # We swap the eyes because the first eye is the right one
                    re_point = Point(
                        x=eyes[0].x + bounding_box.top_left.x,
                        y=eyes[0].y + bounding_box.top_left.y,
                    )
                    le_point = Point(
                        x=eyes[1].x + bounding_box.top_left.x,
                        y=eyes[1].y + bounding_box.top_left.y,
                    )
                    if le_point not in bounding_box or re_point not in bounding_box:
                        le_point = None
                        re_point = None

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=float(confidence),
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            img=img,
            detections=detected_faces,
        )
