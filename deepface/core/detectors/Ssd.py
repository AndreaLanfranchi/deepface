from typing import List

import os
import gdown
import cv2
import pandas
import numpy

from deepface.commons import folder_utils
from deepface.core.detector import Detector as DetectorBase, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger.get_instance()


class Detector(DetectorBase):
    """
    This class is used to detect faces using OpenCV's SSD face detector.
    Note! This is an optional detector, ensure the library is installed.
    """

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._initialize()

    def _initialize(self):

        try:

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

            self._detector = cv2.dnn.readNetFromCaffe(output1, output2)
            self._opencv_detector = self.instance("opencv")

        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.dnn module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

    def process(self, img: numpy.ndarray) -> List[FacialAreaRegion]:
        """
        Detect in picture face(s) with ssd

        Args:
            img (numpy.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """

        results = []
        (h, w) = img.shape[:2]
        if h == 0 or w == 0:
            return results

        detected_face = None
        ssd_labels = [
            "img_id",
            "is_face",
            "confidence",
            "left",
            "top",
            "right",
            "bottom",
        ]

        target_h = 300
        target_w = 300
        aspect_ratio_x = w / target_w
        aspect_ratio_y = h / target_h

        blob = cv2.dnn.blobFromImage(
            image=cv2.resize(img, (target_h, target_w)),
            scalefactor=1.0,
            size=(target_h, target_w),
            mean=(104.0, 177.0, 123.0),
        )
        self._detector.setInput(blob)
        detections = self._detector.forward()

        detections_df = pandas.DataFrame(detections[0][0], columns=ssd_labels)

        detections_df = detections_df[
            detections_df["is_face"] == 1
        ]  # 0: background, 1: face
        detections_df = detections_df[detections_df["confidence"] >= 0.90]

        detections_df["left"] = (detections_df["left"] * 300).astype(int)
        detections_df["bottom"] = (detections_df["bottom"] * 300).astype(int)
        detections_df["right"] = (detections_df["right"] * 300).astype(int)
        detections_df["top"] = (detections_df["top"] * 300).astype(int)

        if detections_df.shape[0] > 0:

            for _, instance in detections_df.iterrows():

                left = instance["left"]
                right = instance["right"]
                bottom = instance["bottom"]
                top = instance["top"]
                confidence = instance["confidence"]

                x = int(left * aspect_ratio_x)
                y = int(top * aspect_ratio_y)
                w = int(right * aspect_ratio_x) - int(left * aspect_ratio_x)
                h = int(bottom * aspect_ratio_y) - int(top * aspect_ratio_y)

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

                left_eye, right_eye = self._opencv_detector.find_eyes(detected_face)

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
