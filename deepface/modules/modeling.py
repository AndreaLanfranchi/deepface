# built-in dependencies
from typing import Any

# project dependencies
from deepface.commons.logger import Logger
from deepface.basemodels import VGGFace, OpenFace, FbDeepFace, DeepID, ArcFace, SFace, Dlib, Facenet
from deepface.extendedmodels import Age, Gender, Race, Emotion

logger = Logger(module="modules.modeling")

def build_model(model_name: str) -> Any:
    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built model class
    """

    # singleton design pattern
    global model_obj

    models = {
        "VGG-Face": VGGFace.VggFaceClient,
        "OpenFace": OpenFace.OpenFaceClient,
        "Facenet": Facenet.FaceNet128dClient,
        "Facenet512": Facenet.FaceNet512dClient,
        "DeepFace": FbDeepFace.DeepFaceClient,
        "DeepID": DeepID.DeepIdClient,
        "Dlib": Dlib.DlibClient,
        "ArcFace": ArcFace.ArcFaceClient,
        "SFace": SFace.SFaceClient,
        "Emotion": Emotion.EmotionClient,
        "Age": Age.ApparentAgeClient,
        "Gender": Gender.GenderClient,
        "Race": Race.RaceClient,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model_obj[model_name] = model()
            logger.info(message=f"Built model : {model_name}")
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]
