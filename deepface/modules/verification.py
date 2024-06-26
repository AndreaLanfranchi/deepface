# built-in dependencies
import time
from typing import Any, Dict, Optional, Union

# 3rd party dependencies
import numpy

# project dependencies
from deepface.infra import detection
from deepface.modules import representation
from deepface.core.extractor import Extractor


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

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        img1_path (str or numpy.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        img2_path (str or numpy.ndarray): Path to the second image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv)

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (bool): Flag to enable face alignment (default is True).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

    Returns:
        result (dict): A dictionary containing verification results.

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

    tic = time.time()

    # --------------------------------
    model: Extractor = Extractor.instance(decomposer)
    target_size = model.input_shape

    # img pairs might have many faces
    img1_objs = detection.detect_faces(
        inp=img1_path,
        target_size=target_size,
        detector=detector,
        grayscale=False,
        align=align,
    )

    img2_objs = detection.detect_faces(
        inp=img2_path,
        target_size=target_size,
        detector=detector,
        grayscale=False,
        align=align,
    )
    # --------------------------------
    distances = []
    regions = []

    distance_metric = distance_metric.lower().strip()
    if distance_metric == "cosine":
        distance_fn = find_cosine_distance
    elif distance_metric == "euclidean":
        distance_fn = find_euclidean_distance
    elif distance_metric == "euclidean_l2":
        distance_fn = find_euclidean_l2_distance
    else:
        raise NotImplementedError("Invalid distance_metric passed : ", distance_metric)

    # now we will find the face pair with minimum distance
    for img1_obj in img1_objs:
        img1_content = img1_obj["face"]
        img1_region = img1_obj["facial_area"]

        img1_embedding_obj = representation.represent(
            img_path=img1_content,
            decomposer=decomposer,
            detector="donotdetect",
            align=align,
            normalization=normalization,
        )
        img1_representation = img1_embedding_obj[0]["embedding"]

        for img2_obj in img2_objs:
            img2_content = img2_obj["face"]
            img2_region = img2_obj["facial_area"]

            img2_embedding_obj = representation.represent(
                img_path=img2_content,
                decomposer=decomposer,
                detector="donotdetect",
                align=align,
                normalization=normalization,
            )

            img2_representation = img2_embedding_obj[0]["embedding"]
            distance = distance_fn(img1_representation, img2_representation)
            distances.append(distance)
            regions.append((img1_region, img2_region))

    # -------------------------------
    threshold = find_threshold(decomposer, distance_metric)
    distance = min(distances)  # best distance
    facial_areas = regions[numpy.argmin(distances)]

    # pylint: disable=simplifiable-if-expression
    resp_obj = {
        "verified": True if distance <= threshold else False,
        "distance": distance,
        "threshold": threshold,
        "model": decomposer,
        "detector_backend": detector,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
        "time": round(time.time() - tic, 2),
    }

    return resp_obj


def find_cosine_distance(
    source_representation: Union[numpy.ndarray, list],
    test_representation: Union[numpy.ndarray, list]
) -> numpy.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (numpy.ndarray or list): 1st vector
        test_representation (numpy.ndarray or list): 2nd vector
    Returns
        distance (numpy.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = numpy.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = numpy.array(test_representation)

    a = numpy.matmul(numpy.transpose(source_representation), test_representation)
    b = numpy.sum(numpy.multiply(source_representation, source_representation))
    c = numpy.sum(numpy.multiply(test_representation, test_representation))
    return 1 - (a / (numpy.sqrt(b) * numpy.sqrt(c)))


def find_euclidean_distance(
    source_representation: Union[numpy.ndarray, list],
    test_representation: Union[numpy.ndarray, list]
) -> numpy.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (numpy.ndarray or list): 1st vector
        test_representation (numpy.ndarray or list): 2nd vector
    Returns
        distance (numpy.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = numpy.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = numpy.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = numpy.sum(numpy.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = numpy.sqrt(euclidean_distance)
    return euclidean_distance

def find_euclidean_l2_distance(
    source_representation: Union[numpy.ndarray, list],
    test_representation: Union[numpy.ndarray, list]
) -> numpy.float64:
    return find_euclidean_distance(
        l2_normalize(source_representation),
        l2_normalize(test_representation)
        )

def l2_normalize(x: Union[numpy.ndarray, list]) -> numpy.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (numpy.ndarray or list): given vector
    Returns:
        y (numpy.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = numpy.array(x)
    return x / numpy.sqrt(numpy.sum(numpy.multiply(x, x)))


def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair
    Args:
        model_name (str): facial recognition model name
        distance_metric (str): distance metric name. Options are cosine, euclidean
            and euclidean_l2.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
        "VGGFace": {
            "cosine": 0.68,
            "euclidean": 1.17,
            "euclidean_l2": 1.17,
        },  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold
