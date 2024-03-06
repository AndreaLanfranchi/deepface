import os
from typing import Any, List
import cv2
import numpy
from deepface.models.Detector import Detector as DetectorBase, FacialAreaRegion


class Detector(DetectorBase):
    """
    This class is used to detect faces using OpenCV face detection.
    """

    def __init__(self):
        super().__init__()
        self._name = "OpenCV"
        self.__initialize()

    def __initialize(self):
        self._detector = self.__build_cascade("haarcascade")
        self._eye_detector = self.__build_cascade("haarcascade_eye")

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with opencv

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        results = []
        if img.shape[0] == 0 or img.shape[1] == 0:
            return results

        detected_face = None

        faces = []
        try:
            # faces = detector["face_detector"].detectMultiScale(img, 1.3, 5)

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

    def __build_cascade(self, model_name="haarcascade") -> Any:
        """
        Build a opencv face&eye detector models
        Returns:
            model (Any)
        """
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
            if os.path.isfile(face_detector_path) != True:
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = opencv_path + "haarcascade_eye.xml"
            if os.path.isfile(eye_detector_path) != True:
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    eye_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(
                f"unimplemented model_name for build_cascade - {model_name}"
            )

        return detector

    def __get_opencv_path(self) -> str:
        """
        Returns where opencv installed
        Returns:
            installation_path (str)
        """
        opencv_home = cv2.__file__
        folders = opencv_home.split(os.path.sep)[0:-1]

        path = folders[0]
        for folder in folders[1:]:
            path = path + "/" + folder

        return path + "/data/"
