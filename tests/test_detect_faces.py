import numpy
import pytest
from deepface import DeepFace
from deepface.commons.logger import Logger
from deepface.core.detector import Detector

logger = Logger.get_instance()

detectors = ["opencv", "mtcnn"]

def test_all_detectors_must_succeed():
    # All detectors must succeed on a well-known image

    result_keys = ["face", "facial_area", "confidence"]
    area_keys = ["x", "y", "w", "h"]

    for detector in Detector.get_available_detectors():
        try:
            detector = Detector.instance(detector)
        except ImportError:
            continue # Don't test unavailable detectors

        results = DeepFace.detect_faces(img_path="dataset/img11.jpg", detector=detector.name)
        assert results is not None and len(results) > 0
        for result in results:
            assert all(key in result for key in result_keys)
            assert all(key in result["facial_area"] for key in area_keys)
            area = result["facial_area"]
            assert area["w"] > 0 and area["h"] > 0

        logger.info(f"✅ detect_faces for {detector.name} backend test is done")


def test_all_detectors_must_faile():
    # All detectors must fail on a well-known image
    black_img = numpy.zeros([224, 224, 3])
    for detector in Detector.get_available_detectors():
        try:
            detector = Detector.instance(detector)                   
        except ImportError:
            continue  # Don't test unavailable detectors

        # DoNotDetect is an exception as it is designed to
        # always return the original image. TODO: remove
        if detector.name == "DoNotDetect":
            continue

        results = DeepFace.detect_faces(
            img_path=black_img, detector=detector.name
        )
        assert results is None or len(results) == 0

        logger.info(f"✅ fail_detect_faces for {detector.name} backend test is done")


# def test_backends_for_enforced_detection_with_non_facial_inputs():
#     black_img = numpy.zeros([224, 224, 3])
#     for detector in detectors:
#         with pytest.raises(ValueError):
#             _ = DeepFace.detect_faces(img_path=black_img, detector=detector)
#     logger.info("✅ detect_faces for enforced detection and non-facial image test is done")


# def test_backends_for_not_enforced_detection_with_non_facial_inputs():
#     black_img = numpy.zeros([224, 224, 3])
#     for detector in detectors:
#         objs = DeepFace.detect_faces(
#             img_path=black_img, detector=detector
#         )
#         assert objs[0]["face"].shape == (224, 224, 3)
#     logger.info("✅ detect_faces for not enforced detection and non-facial image test is done")
