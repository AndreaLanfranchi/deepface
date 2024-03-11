from typing import Any, List

import os
import cv2
import numpy

from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion

# OpenCV's detector (default)
class Detector(DetectorBase):

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):
        self._detector = self._build_cascade("haarcascade")
        self._eye_detector = self._build_cascade("haarcascade_eye")

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:

        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results
        faces = []
        try:
            # note that, by design, opencv's haarcascade scores are >0 but not capped at 1
            faces, _, scores = self._detector.detectMultiScale3(
                img, 1.1, 10, outputRejectLevels=True
            )
        except:
            pass

        if len(faces) > 0:
            for (x, y, w, h), confidence in zip(faces, scores):
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                left_eye, right_eye = self.find_eyes(img=detected_face)
                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=(100 - confidence) / 100,
                )
                results.append(facial_area)

        return results

    def find_eyes(self, img: numpy.ndarray) -> tuple:
        """
        Find the left and right eye coordinates of given image
        Args:
            img (numpy.ndarray): given image
        Returns:
            left and right eye (tuple)
        """
        left_eye = None
        right_eye = None

        # if image has unexpectedly 0 dimension then skip alignment
        if img.shape[0] == 0 or img.shape[1] == 0:
            return left_eye, right_eye

        detected_face_gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )  # eye detector expects gray scale image

        eyes = self._eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

        # ----------------------------------------------------------------

        # opencv eye detection module is not strong. it might find more than 2 eyes!
        # besides, it returns eyes with different order in each call (issue 435)
        # this is an important issue because opencv is the default detector and ssd also uses this
        # find the largest 2 eye. Thanks to @thelostpeace

        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        # ----------------------------------------------------------------
        if len(eyes) >= 2:
            # decide left and right eye

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # -----------------------
            # find center of eyes
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)),
            )
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)),
            )
        return left_eye, right_eye

    def _build_cascade(self, model_name="haarcascade") -> Any:

        match model_name:
            case "haarcascade":
                file_name = "haarcascade_frontalface_default.xml"
            case "haarcascade_eye":
                file_name = "haarcascade_eye.xml"
            case _:
                raise NotImplementedError(f"Unknown : {model_name}")

        cv2_root = os.path.dirname(cv2.__file__)
        file_path = os.path.join(cv2_root, "data", file_name)
        if os.path.isfile(file_path) != True:
            raise RuntimeError(
                f"Coulnd't find {file_path}\n" "Check opencv is installed properly"
            )

        return cv2.CascadeClassifier(file_path)
