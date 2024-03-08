import time

# built-in dependencies
from typing import Any

# project dependencies
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    FbDeepFace,
    DeepID,
    ArcFace,
    SFace,
    Dlib,
    Facenet,
)
from deepface.models.Demography import Demography

from deepface.commons.logger import Logger

logger = Logger(module="modules.modeling")


def get_recognition_model(name: str) -> Any:
    """
    This function retturns a face recognition model.
    Eventually the model instance is lazily initialized.

    Params:
        name (string): The name of the face recognition model to be returned
            Valid values are any of the following:\n
            "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace"

    Exception:
        KeyError: when name is not known

    Returns:
        reference to built model class instance
    """

    name = name.replace(" ", "")
    if len(name) == 0:
        raise KeyError("Empty model name")

    global available_recognition_models
    if not "available_recognition_models" in globals():
        available_recognition_models = {
            "VGG-Face": VGGFace.VggFaceClient,
            "OpenFace": OpenFace.OpenFaceClient,
            "Facenet": Facenet.FaceNet128dClient,
            "Facenet512": Facenet.FaceNet512dClient,
            "DeepFace": FbDeepFace.DeepFaceClient,
            "DeepID": DeepID.DeepIdClient,
            "Dlib": Dlib.DlibClient,
            "ArcFace": ArcFace.ArcFaceClient,
            "SFace": SFace.SFaceClient,
        }

    global recognition_model_instances
    if not "recognition_model_instances" in globals():
        recognition_model_instances = {}

    if not name in recognition_model_instances.keys():
        if not name in available_recognition_models.keys():
            raise KeyError(f"Unknown recognition model : {name}")

        tic = time.time()
        recognition_model_instances[name] = available_recognition_models[name]()
        logger.debug(
            f"Instantiated recognition model : {name} ({time.time() - tic:.3f} seconds)"
        )

    return recognition_model_instances[name]


def get_analysis_model(name: str) -> Demography:
    return Demography.instance(name)
