from typing import List, Optional

import numpy

from deepface.core.detector import Detector as DetectorBase
from deepface.core.types import (
    BoundingBox,
    BoxDimensions,
    DetectedFace,
    Point,
    RangeInt,
)
from deepface.commons.logger import Logger
from deepface.core.exceptions import FaceNotFound, MissingOptionalDependency

try:
    from mediapipe.python.solutions.face_detection import FaceDetection
except ModuleNotFoundError:
    what: str = f"{__name__} requires `mediapipe` library."
    what += "You can install by 'pip install mediapipe' "
    raise MissingOptionalDependency(what) from None

logger = Logger.get_instance()


# MediaPipe detector (optional)
# see also: https://google.github.io/mediapipe/solutions/face_detection
class Detector(DetectorBase):

    _detector: FaceDetection

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        #   min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        #     detection to be considered successful (default 0.5). See details in
        #     https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
        #   model_selection: 0 or 1. 0 to select a short-range model that works
        #     best for faces within 2 meters from the camera, and 1 for a full-range
        #     model best for faces within 5 meters. (default 0) See details in
        #     https://solutions.mediapipe.dev/face_detection#model_selection.
        self._detector = FaceDetection(min_detection_confidence=0.7)

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

        detection_result = self._detector.process(img)
        detections = getattr(detection_result, "detections", [])
        if detections is not None:

            # Extract the bounding box, the landmarks and the confidence score
            for detection in detections:
                if detection is None:
                    continue

                assert len(detection.score) == 1
                confidence = float(round(detection.score[0], 2))
                if min_confidence is not None and confidence < min_confidence:
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
                if x_range.span <= 0 or y_range.span <= 0:
                    continue  # Invalid detection

                if isinstance(min_dims, BoxDimensions):
                    if min_dims.width > 0 and x_range.span < min_dims.width:
                        continue
                    if min_dims.height > 0 and y_range.span < min_dims.height:
                        continue

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

                le_point = None
                re_point = None

                keypoints = detection.location_data.relative_keypoints
                if detect_eyes and keypoints is not None and len(keypoints) == 6:

                    # We swap the left and right eyes: from observer's point of view
                    # to the image's point of view
                    x1 = int(min(round(keypoints[1].x * img_width), img_width))
                    y1 = int(min(round(keypoints[1].y * img_height), img_height))
                    x2 = int(min(round(keypoints[0].x * img_width), img_width))
                    y2 = int(min(round(keypoints[0].y * img_height), img_height))
                    le_point = Point(x1, y1)
                    re_point = Point(x2, y2)

                    # Martian positions ?
                    # TODO Decide whether to discard the face or to not include the eyes
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
            raise FaceNotFound("No face detected. Check the input image.")

        return DetectorBase.Results(
            detector=self.name,
            source=img,
            detections=detected_faces,
        )
