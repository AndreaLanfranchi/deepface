import numpy
import pytest
from deepface import DeepFace
from deepface.commons.logger import Logger
from deepface.core.detector import Detector

logger = Logger.get_instance()

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

        logger.info(f"✅ Detector {detector.name} succeeded")


def test_all_detectors_must_fail():
    # All detectors must fail on a black image
    black_img = numpy.zeros([224, 224, 3], dtype=numpy.uint8)
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
        logger.info(f"✅ Detector {detector.name} successfully failed")