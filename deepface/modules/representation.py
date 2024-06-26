from typing import Any, Dict, Optional, List, Union

import numpy
import cv2

from deepface.infra import detection
from deepface.core import imgutils
from deepface.core.extractor import Extractor


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
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'.

        align (boolean): Perform alignment based on the eye positions.

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

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
    resp_objs = []

    model: Extractor = Extractor.instance(decomposer)

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = model.input_shape
    if detector != "donotdetect":
        img_objs = detection.detect_faces(
            inp=img_path,
            target_size=(target_size[1], target_size[0]),
            detector=detector,
            grayscale=False,
            align=align,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = imgutils.load_image(img_path)
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, (target_size.width, target_size.height))
            img = numpy.expand_dims(img, axis=0)
            # when called from verify, this is already normalized. But needed when user given.
            if img.max() > 1:
                img = (img.astype(numpy.float32) / 255.0).astype(numpy.float32)
        # --------------------------------
        # make dummy region and confidence to keep compatibility with `detect_faces`
        img_objs = [
            {
                "face": img,
                "facial_area": {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]},
                "confidence": 0,
            }
        ]
    # ---------------------------------

    for img_obj in img_objs:
        img = img_obj["face"]
        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]
        # custom normalization
        img = imgutils.normalize_input(img=img, mode=normalization)

        embedding = model.process(img)

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs
