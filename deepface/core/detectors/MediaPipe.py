from typing import Dict, List, Optional

import numpy

from deepface.core.detector import Detector as DetectorBase
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)
from deepface.core.exceptions import FaceNotFoundError, MissingDependencyError
from deepface.commons.logger import Logger

logger = Logger.get_instance()

try:
    from mediapipe.python.solutions.face_detection import FaceDetection
except ModuleNotFoundError:
    what: str = f"{__name__} requires `mediapipe` library."
    what += "You can install by 'pip install mediapipe' "
    raise MissingDependencyError(what) from None


# MediaPipe detector (optional)
# see also: https://google.github.io/mediapipe/solutions/face_detection
class Detector(DetectorBase):

    _detector: FaceDetection

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._KDEFAULT_MIN_CONFIDENCE = float(0.7)
        self._initialize()

    def _initialize(self):
        
        #   min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        #     detection to be considered successful (default 0.5). See details in
        #     https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
        #   model_selection: 0 or 1. 0 to select a short-range model that works
        #     best for faces within 2 meters from the camera, and 1 for a full-range
        #     model best for faces within 5 meters. (default 0) See details in
        #     https://solutions.mediapipe.dev/face_detection#model_selection.
        
        self._detector = FaceDetection(
            min_detection_confidence=self._KDEFAULT_MIN_CONFIDENCE,
            model_selection=1,
        )

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

        img_height, img_width = img.shape[:2]
        detected_faces: List[DetectedFace] = []

        detection_result = self._detector.process(img)
        detections = getattr(detection_result, "detections", [])
        if detections is not None:

            # Extract the bounding box, the landmarks and the confidence score
            for detection in detections:
                if detection is None:
                    continue

                assert len(detection.score) == 1
                confidence = round(float(detection.score[0]), 5)
                if confidence < min_confidence:
                    continue

                rbb = detection.location_data.relative_bounding_box
                x_range = RangeInt(
                    int(round(rbb.xmin * img_width)),
                    int(round(rbb.xmin * img_width)) + int(round(rbb.width * img_width)),
                )
                y_range = RangeInt(
                    int(round(rbb.ymin * img_height)),
                    int(round(rbb.ymin * img_height)) + int(round(rbb.height * img_height)),
                )
                if x_range.span <= min_dims.width or y_range.span <= min_dims.height:
                    continue  # Invalid or empty detection

                bounding_box: BoundingBox = BoundingBox(
                    top_left=Point(x=x_range.start, y=y_range.start),
                    bottom_right=Point(x=x_range.end, y=y_range.end),
                )

                # pylint: disable=line-too-long
                # See also: https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
                # 6 [0,5] approximate facial keypoint coordinates:
                # 0 Left eye (from the observer’s point of view)
                # 1 Right eye
                # 2 Nose tip
                # 3 Mouth
                # 4 Left eye tragion
                # 5 Right eye tragion
                # pylint: enable=line-too-long

                points: Optional[Dict[str, Point]] = None
                relative_keypoints = detection.location_data.relative_keypoints
                if key_points and relative_keypoints is not None and len(relative_keypoints) > 0:
                    points = dict[str, Point]()
                    if len(relative_keypoints) >= 2:
                        x1 = int(min(round(relative_keypoints[1].x * img_width), img_width))
                        y1 = int(min(round(relative_keypoints[1].y * img_height), img_height))
                        x2 = int(min(round(relative_keypoints[0].x * img_width), img_width))
                        y2 = int(min(round(relative_keypoints[0].y * img_height), img_height))
                        le_point = Point(x1, y1)
                        re_point = Point(x2, y2)
                        points = {"lec": le_point, "rec": re_point}

                    if len(relative_keypoints) >= 3:
                        x = int(min(round(relative_keypoints[2].x * img_width), img_width))
                        y = int(min(round(relative_keypoints[2].y * img_height), img_height))
                        n_point = Point(x, y)
                        points.update({"nt": n_point})

                    if len(relative_keypoints) >= 4:
                        x = int(min(round(relative_keypoints[3].x * img_width), img_width))
                        y = int(min(round(relative_keypoints[3].y * img_height), img_height))
                        cm_point = Point(x, y)
                        points.update({"mc": cm_point})

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
