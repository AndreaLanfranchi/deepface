from typing import Any, List

import numpy

from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion


class Detector(DetectorBase):
    """
    This class is used to detect faces using fast mtcnn face detector.
    Note! This is an optional detector, ensure the library is installed.

    See also:
    https://google.github.io/mediapipe/solutions/face_detection
    """

    _detector: Any

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self.__initialize()

    def __initialize(self):
        try:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection
            self._detector = mp_face_detection.FaceDetection(
                min_detection_confidence=0.7
            )

        except ModuleNotFoundError as e:
            raise ImportError(
                "MediaPipe is an optional detector, ensure the library is installed."
                "Please install using 'pip install mediapipe' "
            ) from e

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with mediapipe

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        img_width = img.shape[1]
        img_height = img.shape[0]
        results = self._detector.process(img)

        # If no face has been detected, return an empty list
        if results.detections is None:
            return results

        # Extract the bounding box, the landmarks and the confidence score
        for current_detection in results.detections:
            (confidence,) = current_detection.score

            bounding_box = current_detection.location_data.relative_bounding_box
            landmarks = current_detection.location_data.relative_keypoints

            x = int(bounding_box.xmin * img_width)
            w = int(bounding_box.width * img_width)
            y = int(bounding_box.ymin * img_height)
            h = int(bounding_box.height * img_height)

            left_eye = (
                int(landmarks[0].x * img_width),
                int(landmarks[0].y * img_height),
            )
            right_eye = (
                int(landmarks[1].x * img_width),
                int(landmarks[1].y * img_height),
            )
            # nose = (int(landmarks[2].x * img_width), int(landmarks[2].y * img_height))
            # mouth = (int(landmarks[3].x * img_width), int(landmarks[3].y * img_height))
            # right_ear = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
            # left_ear = (int(landmarks[5].x * img_width), int(landmarks[5].y * img_height))

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            results.append(facial_area)

        return results