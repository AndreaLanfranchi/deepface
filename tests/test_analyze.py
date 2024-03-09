import cv2
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger.get_instance()

detectors = ["opencv", "mtcnn"]


def test_standard_analyze():
    img = "dataset/img4.jpg"
    results = DeepFace.analyze(img)
    for result in results:
        logger.debug(result)
        assert result["age"] > 20 and result["age"] < 40
        assert result["gender"] == "Female"
    logger.info("✅ test standard analyze done")


def test_analyze_with_all_actions_as_tuple():
    img = "dataset/img4.jpg"
    attributes = ("age", "gender", "race", "emotion")
    results = DeepFace.analyze(img, attributes=attributes)

    for result in results:
        for attr in attributes:
            assert result.get(attr) is not None
            logger.debug(f"{attr}: {result[attr]}")

    logger.info("✅ test analyze for all actions as tuple done")


def test_analyze_with_all_actions_as_list():
    img = "dataset/img4.jpg"
    attributes = ["age", "gender", "race", "emotion"]
    results = DeepFace.analyze(img, attributes=attributes)

    for result in results:
        for attr in attributes:
            assert result.get(attr) is not None
            logger.debug(f"{attr}: {result[attr]}")

    logger.info("✅ test analyze for all actions as list done")


def test_analyze_for_some_actions():
    img = "dataset/img4.jpg"
    attributes = ["age", "gender"]
    results = DeepFace.analyze(img, attributes=attributes)

    for result in results:
        for attr in attributes:
            assert result.get(attr) is not None
            logger.debug(f"{attr}: {result[attr]}")
        # these are not requested attributes
        assert result.get("race") is None
        assert result.get("emotion") is None

    logger.info("✅ test analyze for some actions done")


def test_analyze_for_preloaded_image():
    img = cv2.imread("dataset/img1.jpg")
    results = DeepFace.analyze(img)
    for result in results:
        logger.debug(result)
        assert result["age"] > 20 and result["age"] < 40
        assert result["gender"] == "Female"

    logger.info("✅ test analyze for pre-loaded image done")


def test_analyze_for_different_detectors():
    images = [
        "dataset/img1.jpg",
        "dataset/img5.jpg",
        "dataset/img6.jpg",
        "dataset/img8.jpg",
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img6.jpg",
        "dataset/img6.jpg",
    ]

    for image in images:
        for detector in detectors:
            results = DeepFace.analyze(
                image,
                attributes=("gender",),
                attributes_details=True,
                detector_backend=detector,
            )
            for result in results:
                logger.debug(result)
                attr_gender = result.get("gender")
                assert attr_gender is not None
                assert attr_gender in ["Female", "Male"]

                analysis = result.get("gender_analysis")
                assert analysis is not None

                analysis_female = analysis.get("Female")
                analysis_male = analysis.get("Male")
                assert analysis_female is not None
                assert analysis_male is not None

                if attr_gender == "Male":
                    assert analysis_male > analysis_female
                else:
                    assert analysis_female > analysis_male

    logger.info("✅ test analyze for different detectors done")
