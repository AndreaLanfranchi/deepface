from typing import List, Union, Optional

import numpy

from deepface.modules import preprocessing
from deepface.core.detector import Detector
from deepface.commons.logger import Logger

logger = Logger.get_instance()

def detect_faces(
    img: Union[str, numpy.ndarray],
    detector: Union[str, Detector] = Detector.get_default(),
) -> Detector.Results:

    results: List[Detector.Results] = []
    if not isinstance(detector, Detector):
        detector = Detector.instance(detector)

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img, _ = preprocessing.load_image(img)
    result: Detector.Results = detector.process(img)
    if len(result) != 0:
        results.append(result)

    return results
