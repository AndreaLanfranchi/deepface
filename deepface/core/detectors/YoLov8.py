from typing import Dict, List, Optional

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
        self._KDEFAULT_MIN_CONFIDENCE = float(0.8)
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
            if item.boxes is None:
                continue

            confidence = round(float(item.boxes.conf.tolist()[0]), 5)
            if confidence < min_confidence:
                continue

            # Extract the bounding box as integers
            # YoLov8 returns the center of the box along with width and height
            # as a consequence we need to calculate the top-left corner
            x, y, w, h = item.boxes.xywh.tolist()[0]
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            x_range = RangeInt(x, min(x + w, img_width))
            y_range = RangeInt(y, min(y + h, img_height))
            if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                continue  # Invalid or empty detection

            bounding_box = BoundingBox(
                top_left=Point(x=x_range.start, y=y_range.start),
                bottom_right=Point(x=x_range.end, y=y_range.end),
            )

            points: Optional[Dict[str, Optional[Point]]] = None
            if key_points and item.keypoints is not None:

                # Indices (note we have to swap from viewer to image perspective):
                # 0: left eye (from the viewer's perspective)
                # 1: right eye (from the viewer's perspective)
                # 2: nose tip
                # 3: left corner of the mouth (from the viewer's perspective)
                # 4: right corner of the mouth (from the viewer's perspective)
                for i in range(0, len(item.keypoints.xy[0])):
                    xy = item.keypoints.xy[0][i].tolist()

                    xy_point = Point(x=int(round(xy[0])), y=int(round(xy[1])))
                    if xy_point not in bounding_box:
                        continue

                    xy_key: Optional[str] = None
                    match i:
                        case 0:
                            xy_key = "rec"
                        case 1:
                            xy_key = "lec"
                        case 2:
                            xy_key = "nt"
                        case 3:
                            xy_key = "mrc"
                        case 4:
                            xy_key = "mlc"
                        case _:
                            # should not happen
                            pass

                    if xy_key is not None:
                        if points is None:
                            points = {}
                        points[xy_key] = xy_point

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
