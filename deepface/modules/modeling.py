# built-in dependencies
from typing import Any

# project dependencies
from deepface.commons.logger import Logger
from deepface.basemodels import VGGFace, OpenFace, FbDeepFace, DeepID, ArcFace, SFace, Dlib, Facenet
from deepface.extendedmodels import Age, Gender, Race, Emotion

logger = Logger(module="modules.modeling")

def build_model(
        model_name: str,
        silent: bool = False
        ) -> Any:
    """
    This function builds a deepface model

    Params:
        model_name (string): face recognition or facial attribute model

            For face recognition, available models are:\n
            "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace"

            For facial attribute analysis, available models are:\n
            "Age", "Gender", "Emotion", "Race"
        
        silent (bool): whether to print logs or not
    
    Exception:
        KeyError: if model_name is not in available_models

    Returns:
        reference to built model class instance
    """

    global available_models
    if not "available_models" in globals():
        available_models = {
        # Face recognition models
        "VGG-Face": VGGFace.VggFaceClient,
        "OpenFace": OpenFace.OpenFaceClient,
        "Facenet": Facenet.FaceNet128dClient,
        "Facenet512": Facenet.FaceNet512dClient,
        "DeepFace": FbDeepFace.DeepFaceClient,
        "DeepID": DeepID.DeepIdClient,
        "Dlib": Dlib.DlibClient,
        "ArcFace": ArcFace.ArcFaceClient,
        "SFace": SFace.SFaceClient,
        
        # Facial attribute analysis models
        "Emotion": Emotion.EmotionClient,
        "Age": Age.ApparentAgeClient,
        "Gender": Gender.GenderClient,
        "Race": Race.RaceClient
        }

    global model_instances
    if not "model_instances" in globals():
        model_instances = {}

    if not model_name in model_instances.keys():
        if not model_name in available_models.keys():
            raise KeyError(f"Unknown model_name : - {model_name}")
        if not silent:
            logger.info(f"Instantiating model : {model_name}")
        model_instances[model_name] = available_models[model_name]()

    return model_instances[model_name]
