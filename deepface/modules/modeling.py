# built-in dependencies
from typing import Any

# project dependencies
from deepface.commons.logger import Logger
from deepface.basemodels import VGGFace, OpenFace, FbDeepFace, DeepID, ArcFace, SFace, Dlib, Facenet
from deepface.extendedmodels import Age, Gender, Race, Emotion

logger = Logger(module="modules.modeling")

def get_recognition_model(
        name: str,
        silent: bool = False
        ) -> Any:
    """
    This function retturns a face recognition model.
    Eventually the model instance is lazily initialized.

    Params:
        name (string): The name of the face recognition model to be returned
            Valid values are any of the following:\n
            "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace"
        
        silent (bool): whether to print logs or not
    
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
        if not silent:
            logger.info(f"Instantiating recognition model : {name}")
        recognition_model_instances[name] = available_recognition_models[name]()

    return recognition_model_instances[name]

def get_analysis_model(
        name: str,
        silent: bool = False
        ) -> Any:
    """
    This function retturns a face analisys model.
    Eventually the model instance is lazily initialized.

    Params:
        name (string): The name of the face analisys model to be returned
            Valid values are any of the following:\n
            "Age", "Gender", "Emotion", "Race"
        
        silent (bool): whether to print logs or not
    
    Exception:
        KeyError: when name is not known

    Returns:
        reference to built model class instance
    """

    name = name.replace(" ", "")
    if len(name) == 0:
        raise KeyError("Empty model name")

    global available_analisys_models
    if not "available_analisys_models" in globals():
        available_analisys_models = {
        "Emotion": Emotion.EmotionClient,
        "Age": Age.ApparentAgeClient,
        "Gender": Gender.GenderClient,
        "Race": Race.RaceClient
        }

    global analisys_model_instances
    if not "analysis_model_instances" in globals():
        analisys_model_instances = {}

    if not name in analisys_model_instances.keys():
        if not name in available_analisys_models.keys():
            raise KeyError(f"Unknown analisys model : {name}")
        if not silent:
            logger.info(f"Instantiating analisys model : {name}")
        analisys_model_instances[name] = available_analisys_models[name]()

    return analisys_model_instances[name]
