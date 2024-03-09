# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy
from tqdm import tqdm
from deepface.core.analyzer import Analyzer

# project dependencies
from deepface.modules import detection


def analyze(
    img_path: Union[str, numpy.ndarray],
    attributes: Union[str, tuple, list] = ("emotion", "age", "gender", "race"),
    attributes_details: bool = False,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
) -> List[Dict[str, Any]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.

    Args:
        img_path (str or numpy.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        attributes (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        attributes_details (bool): Whether to include the details of estimation for each attribute.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
           the analysis results for a detected face.

           Each dictionary in the list contains the following keys:

           - 'region' (dict): Represents the rectangular region of the detected face in the image.
               - 'x': x-coordinate of the top-left corner of the face.
               - 'y': y-coordinate of the top-left corner of the face.
               - 'w': Width of the detected face region.
               - 'h': Height of the detected face region.

           - 'age' (float): Estimated age of the detected face.

           - 'face_confidence' (float): Confidence score for the detected face.
                Indicates the reliability of the face detection.

           - 'dominant_gender' (str): The dominant gender in the detected face.
                Either "Man" or "Woman."

           - 'gender' (dict): Confidence scores for each gender category.
               - 'Man': Confidence score for the male gender.
               - 'Woman': Confidence score for the female gender.

           - 'dominant_emotion' (str): The dominant emotion in the detected face.
                Possible values include "sad," "angry," "surprise," "fear," "happy,"
                "disgust," and "neutral."

           - 'emotion' (dict): Confidence scores for each emotion category.
               - 'sad': Confidence score for sadness.
               - 'angry': Confidence score for anger.
               - 'surprise': Confidence score for surprise.
               - 'fear': Confidence score for fear.
               - 'happy': Confidence score for happiness.
               - 'disgust': Confidence score for disgust.
               - 'neutral': Confidence score for neutrality.

           - 'dominant_race' (str): The dominant race in the detected face.
                Possible values include "indian," "asian," "latino hispanic,"
                "black," "middle eastern," and "white."

           - 'race' (dict): Confidence scores for each race category.
               - 'indian': Confidence score for Indian ethnicity.
               - 'asian': Confidence score for Asian ethnicity.
               - 'latino hispanic': Confidence score for Latino/Hispanic ethnicity.
               - 'black': Confidence score for Black ethnicity.
               - 'middle eastern': Confidence score for Middle Eastern ethnicity.
               - 'white': Confidence score for White ethnicity.
    """

    # validate actions
    if isinstance(attributes, str):
        attributes = (attributes,)

    # check if actions is not an iterable or empty.
    if not hasattr(attributes, "__getitem__") or not attributes:
        raise ValueError("`attributes` must be a list of strings.")

    attributes = list(attributes)

    # For each action, check if it is valid
    for attribute in attributes:
        _ = Analyzer.instance(attribute)

    # ---------------------------------
    resp_objects = []

    img_objs = detection.detect_faces(
        source=img_path,
        target_size=(224, 224),
        detector=detector_backend,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
    )

    for img_obj in img_objs:
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

            resp_objects.append(obj)

    return resp_objects
