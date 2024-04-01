from typing import Any, Dict, List, Tuple, Union, Optional

import os
import sys
import warnings
import logging
import numpy
import pandas
import tensorflow

# package dependencies
from deepface.core.analyzer import Analyzer
from deepface.core.detector import Detector
from deepface.core.extractor import Extractor


from deepface.commons.logger import Logger
from deepface.modules import (
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
)
from deepface import __version__

tensorflow.get_logger().setLevel(logging.ERROR)
logger = Logger.get_instance()

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# -----------------------------------


def verify(
    img1_path: Union[str, numpy.ndarray],
    img2_path: Union[str, numpy.ndarray],
    decomposer: Optional[str] = None,
    detector: Optional[str] = None,
    distance_metric: str = "cosine",
    align: bool = True,
    normalization: str = "base",
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.
    Args:
        img1_path (str or numpy.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        img2_path (str or numpy.ndarray): Path to the second image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (bool): Flag to enable face alignment (default is True).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

    Returns:
        result (dict): A dictionary containing verification results with following keys.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'max_threshold_to_verify' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'similarity_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images.
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    return verification.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        decomposer=decomposer,
        detector=detector,
        distance_metric=distance_metric,
        align=align,
        normalization=normalization,
    )


def analyze(
    img_path: Union[str, numpy.ndarray],
    attributes: Optional[Union[tuple, list]] = None,
    attributes_details: bool = False,
    detector: Optional[str] = None,
    align: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race from the faces detected
    in the provided image (if any). If the source image contains multiple faces, the result will
    include attribute analysis result(s) for each detected face.

    Args:
        img_path (str or numpy.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        attributes (tuple / list): An optional list of facial attributes to analyze. If not provided,
            by default all available attributes analyzers will be used. For a complete list of
            available attribute analyzers, refer to the list of modules in the
            `deepface.core.analyzers` package.

        attributes_details (bool): Whether to include the details of estimation for each attribute.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
           the analysis results for a detected face.

           Each dictionary in the list contains the following keys:

           - 'region' (dict): Represents the rectangular region of the detected face in the image.
               - 'x': x-coordinate of the top-left corner of the face.
               - 'y': y-coordinate of the top-left corner of the face.
               - 'w': Width of the detected face region.
               - 'h': Height of the detected face region.

           - '<attribute_name>' (str): The name of the attribute being analyzed.
                - '<attribute_value>': The most relevant or dominant value of the attribute.

           - '<attribute_name>_analysis' (dict): The detailed analysis of the attribute.
            (if requested). On behalf of the attribute name each possible value and its
            weight (normalized to 100) is provided.

    Raises:
        Any exceptions raised by the face detection and attribute analysis modules.
    """
    return demography.analyze(
        img_path=img_path,
        attributes=attributes,
        attributes_details=attributes_details,
        detector=detector,
        align=align,
    )


def find(
    img: Union[str, numpy.ndarray],
    db_path: str,
    detector: Optional[Union[str, Detector]] = None,
    extractor: Optional[Union[str, Extractor]] = None,
    distance_metric: str = "cosine",
    align: bool = True,
    threshold: Optional[float] = None,
    normalization: str = "base",
) -> List[pandas.DataFrame]:
    """
    Identify individuals in a database
    """
    return recognition.find(
        img=img,
        db_path=db_path,
        detector=detector,
        extractor=extractor,
        distance_metric=distance_metric,
        align=align,
        threshold=threshold,
        normalization=normalization,
    )


def represent(
    img_path: Union[str, numpy.ndarray],
    decomposer: Optional[str] = None,
    detector: Optional[str] = None,
    align: bool = True,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or numpy.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face.).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
            (default is base).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).

        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.

        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
    """
    return representation.represent(
        img_path=img_path,
        decomposer=decomposer,
        detector=detector,
        align=align,
        normalization=normalization,
    )


def stream(
    db_path: str,
    detector: Optional[str] = None,
    extractor: Optional[str] = None,
    distance_metric: str = "cosine",
    analyzers: List[str] = [],
    source: Union[str, int] = int(0),
    freeze_time_seconds: int = 3,
    valid_frames_count: int = 5,
    faces_count_threshold: int = sys.maxsize,
) -> None:
    """
    Run real time face recognition and facial attribute analysis

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        analyzers (List[str]): List of face analyzers to be used. Default is
        ["Age", "Emotion", "Gender"]

        source (Any): The source for the video stream (default is 0, which represents the
            default camera).

        freeze_time_seconds (int): How much time (seconds) to freeze the captured face(s)
            which receive matches from find function (default is 3). This is useful to
            allow operators to visualize the detected face(s) and the matching results.

        valid_frames_count (int): The number of continuos valid frames to be considered as a
            positive face recognition. A valid frame is a frame that contains
            at least one face. Valid value in range of [1, 5] (default is 5).

        faces_count_threshold (int): The maximum number of faces to be detected in a frame.
            If the number of detected faces exceeds this threshold, the frame will be skipped
            for face recognition (default is 1).

    Returns:
        None
    """

    freeze_time_seconds = max(1, freeze_time_seconds)
    valid_frames_count = max(1, valid_frames_count)
    faces_count_threshold = max(1, faces_count_threshold)

    streaming.analysis(
        db_path=db_path,
        extractor=extractor,
        detector=detector,
        distance_metric=distance_metric,
        attributes=analyzers,
        source=source,
        freeze_time_seconds=freeze_time_seconds,
        valid_frames_count=valid_frames_count,
        faces_count_threshold=faces_count_threshold,
    )


def detect_faces(
    img_path: Union[str, numpy.ndarray],
    target_size: Optional[Tuple[int, int]] = None,
    detector: Optional[str] = None,
    align: bool = True,
    grayscale: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image

    Args:
        img_path (str or numpy.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image (default is (224, 224)).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        align (bool): Flag to enable face alignment (default is True).

        grayscale (boolean): Flag to convert the image to grayscale before
            processing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (numpy.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values

        - "confidence" (float): The confidence score associated with the detected face.
    """

    return detection.detect_faces(
        img=img_path,
        target_size=target_size,
        detector=detector,
        align=align,
        grayscale=grayscale,
        human_readable=True,
    )


def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()
