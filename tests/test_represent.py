import cv2
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger.get_instance()


def test_standard_represent():
    img_path = "dataset/img1.jpg"
    embedding_objs = DeepFace.represent(img_path)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        logger.debug(f"Function returned {len(embedding)} dimensional vector")
        assert len(embedding) == 4096
    logger.info("✅ test standard represent function done")


def test_represent_for_skipped_detector_backend_with_image_path():
    face_img = "dataset/img5.jpg"

    result_keys = ["embedding", "facial_area", "face_confidence"]
    area_keys = ["x", "y", "w", "h"]

    results = DeepFace.represent(img_path=face_img, detector="donotdetect")
    assert len(results) == 1

    result = results[0]
    assert all(key in result for key in result_keys)
    assert all(key in result["facial_area"] for key in area_keys)

    logger.info(
        "✅ test represent function for skipped detector and image path input backend done"
    )


def test_represent_for_skipped_detector_backend_with_preloaded_image():
    face_img = "dataset/img5.jpg"
    img = cv2.imread(face_img)

    result_keys = ["embedding", "facial_area", "face_confidence"]
    area_keys = ["x", "y", "w", "h"]

    results = DeepFace.represent(img_path=img, detector="donotdetect")
    assert len(results) == 1

    result = results[0]
    assert all(key in result for key in result_keys)
    assert all(key in result["facial_area"] for key in area_keys)

    logger.info(
        "✅ test represent function for skipped detector and preloaded image done"
    )
