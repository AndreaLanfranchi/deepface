from typing import List, Optional

import os
import numpy
import gdown

from deepface.core.detector import Detector as DetectorBase
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.core.exceptions import FaceNotFoundError, MissingDependencyError
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ModuleNotFoundError:
    what: str = f"{__name__} requires `ultralytics` library."
    what += "You can install by 'pip install ultralytics' "
    raise MissingDependencyError(what) from None

logger = Logger.get_instance()


# YoloV8 detector (optional)
class Detector(DetectorBase):

    _detector: YOLO
    # _LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):

        file_name = "yolov8n-face.pt"
        weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

        if not os.path.isfile(weight_file):
            logger.info(f"Download : {file_name}")
            try:
                url: str = (
                    "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
                )
                gdown.download(url, weight_file, quiet=False, user_agent="Mozilla/5.0")
            except Exception as err:
                raise ValueError(
                    f"Exception while downloading Yolo weights from {url}."
                    f"You may consider to download it to {weight_file} manually."
                ) from err

        self._detector = YOLO(weight_file)

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

        # TODO: ensure we pass a single image otherwise
        # the list will return one element per image
        images_results: List[Results] = self._detector.predict(
            img,
            verbose=False,
            show=False,
            conf=0.25,
        )

        assert (
            len(images_results) == 1
        ), "YOLOv8 should return a single Results object per image"
        results: Results = images_results[0]

        # See https://docs.ultralytics.com/modes/predict/#working-with-results
        # This is extremely counter-intuitive, as the Results object is designed
        # to be iterable returning a new instance of Results for each item

        for item in results:
            if item.boxes is None or item.keypoints is None:
                continue

            confidence = float(item.boxes.conf.tolist()[0])
            if min_confidence is not None and confidence < min_confidence:
                continue

            # Extract the bounding box as integers
            # YoLov8 returns the center of the box along with width and height
            # as a consequence we need to calculate the top-left corner
            x, y, w, h = item.boxes.xywh.tolist()[0]
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            x_range = RangeInt(x, min(x + w, img_width))
            y_range = RangeInt(y, min(y + h, img_height))
            if x_range.span <= 0 or y_range.span <= 0:
                continue  # Invalid detection
            if min_dims is not None:
                if min_dims.width > 0 and x_range.span < min_dims.width:
                    continue
                if min_dims.height > 0 and y_range.span < min_dims.height:
                    continue

            bounding_box = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            le_point = None
            re_point = None
            if detect_eyes:
                left_eye = tuple(int(i) for i in item.keypoints.xy[0][0].tolist())
                right_eye = tuple(int(i) for i in item.keypoints.xy[0][1].tolist())
                le_point = Point(x=left_eye[0], y=left_eye[1])
                re_point = Point(x=right_eye[0], y=right_eye[1])
                if le_point not in bounding_box or re_point not in bounding_box:
                    le_point = None
                    re_point = None
                else:
                    # Is not granted the order in which the eyes are returned
                    if le_point.x < re_point.x:
                        le_point, re_point = re_point, le_point

            detected_faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    left_eye=le_point,
                    right_eye=re_point,
                    confidence=confidence,
                )
            )

        if len(detected_faces) == 0 and raise_notfound == True:
            raise FaceNotFoundError("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            img=img,
            detections=detected_faces,
        )
