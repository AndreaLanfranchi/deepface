from typing import Dict, List, Optional

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
        self._KDEFAULT_MIN_CONFIDENCE = float(0.95)
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
        img_height, img_width = img.shape[:2]

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

            confidence = round(float(row["confidence"]), 5)
            if min_confidence is not None and confidence < min_confidence:
                continue

            x1 = int(round(row["left"] * aspect_ratio_x))
            x2 = int(round(row["right"] * aspect_ratio_x))
            y1 = int(round(row["top"] * aspect_ratio_y))
            y2 = int(round(row["bottom"] * aspect_ratio_y))
            x_range = RangeInt(x1, min(x2, img_width))
            y_range = RangeInt(y1, min(y2, img_height))
            if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                continue  # Invalid or empty detection

            bounding_box: BoundingBox = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            points: Optional[Dict[str, Optional[Point]]] = None
            if key_points:
                cropped_img = img[
                    bounding_box.top_left.y : bounding_box.bottom_right.y,
                    bounding_box.top_left.x : bounding_box.bottom_right.x,
                ]
                eyes: List[Point] = self._opencv_detector.find_eyes(img=cropped_img)
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
                    points = {"lec": le_point, "rec": re_point}

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
