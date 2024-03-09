from deepface import DeepFace

# pylint: disable=broad-except


def represent(img_path, model_name, detector_backend, align):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            decomposer=model_name,
            detector=detector_backend,
            align=align,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        return {"error": f"Exception while representing: {str(err)}"}, 400


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, align
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            decomposer=model_name,
            detector=detector_backend,
            distance_metric=distance_metric,
            align=align
        )
        return obj
    except Exception as err:
        return {"error": f"Exception while verifying: {str(err)}"}, 400


def analyze(img_path, actions, detector_backend, align):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            attributes=actions,
            detector=detector_backend,
            align=align
        )
        result["results"] = demographies
        return result
    except Exception as err:
        return {"error": f"Exception while analyzing: {str(err)}"}, 400
