from typing import List, Optional, Union

import os
import numpy
from tqdm import tqdm

from deepface.core import imgutils
from deepface.core.detector import Detector
from deepface.core.extractor import Extractor
from deepface.core.types import BoxDimensions


def extract_faces(
    inp: Union[str, numpy.ndarray],
    tag: Optional[str] = None,
    detector: Optional[Union[str, Detector]] = None,
    extractor: Optional[Union[str, Extractor]] = None,
    min_dims: Optional[BoxDimensions] = None,
    min_confidence: Optional[float] = None,
    key_points: bool = False,
    raise_notfound: bool = False,
) -> List[float]:
    """
    Extract faces from an image

    Args:
    -----
        input: image path or numpy array
        detector: detector instance or string
        extractor: extractor instance or string
        min_dims: minimum dimensions for detected faces
        raise_notfound: raise exception if no faces are detected

    Returns:
    --------
        extraction results

    Raises:
    -------
        ValueError: if the input is invalid
        Any other exceptions raised by the detector or image loading
        functions
    """

    img, tag = imgutils.load_image(inp, tag=tag)

    detector_instance = Detector.instance(detector)
    detector_results: Detector.Results = detector_instance.process(
        img=img,
        tag=tag,
        min_dims=min_dims,
        min_confidence=min_confidence,
        key_points=key_points, 
        raise_notfound=raise_notfound,
    )
    if not detector_results:
        return []

    extractor_instance = Extractor.instance(extractor)
    for detection in detector_results.detections:
        embedding = extractor_instance.process(img, detection.bounding_box)
        print(embedding)

    return []
