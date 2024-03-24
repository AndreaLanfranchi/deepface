# built-in dependencies
from typing import Any, Dict, List, Optional, Union

# 3rd party dependencies
import numpy
from tqdm import tqdm
from deepface.core.analyzer import Analyzer

# project dependencies
from deepface.modules import detection
from deepface.commons.logger import Logger

logger = Logger.get_instance()


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

    # Assume defaults if not provided
    if attributes is None:
        attributes = Analyzer.get_available_attributes()
    elif isinstance(attributes, str):
        attributes = attributes.lower().strip()
        if attributes in ["", "all", "full", "any", "*"]:
            attributes = Analyzer.get_available_attributes()
        elif "," in attributes:
            attributes = [attribute.strip() for attribute in attributes.split(",")]
        elif " " in attributes:
            attributes = [attribute.strip() for attribute in attributes.split(".")]
        elif ";" in attributes:
            attributes = [attribute.strip() for attribute in attributes.split(";")]
        else:
            attributes = (attributes,)

    # check if actions is not an iterable or empty.
    if not hasattr(attributes, "__getitem__") or not attributes:
        raise ValueError("`attributes` must be a list of strings.")

    attributes = list(attributes)
    results = []

    detected_faces = detection.detect_faces(
        source=img_path,
        target_size=(224, 224),
        detector=detector,
        grayscale=False,
        align=align,
    )

    for img_obj in detected_faces:
        img_content = img_obj["face"]
        img_region = img_obj["facial_area"]
        img_confidence = img_obj["confidence"]
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {}
            # facial attribute analysis
            pbar = tqdm(range(len(attributes)), desc="Analyzing attributes")
            for index in pbar:
                attribute = attributes[index]
                analyzer: Analyzer = Analyzer.instance(attribute)
                pbar.set_description(f"Attribute: {analyzer.name}")
                analysis_result = analyzer.process(
                    img=img_content, detail=attributes_details
                )
                obj.update(analysis_result)
                obj["region"] = img_region
                obj["face_confidence"] = img_confidence

            results.append(obj)

    return results
