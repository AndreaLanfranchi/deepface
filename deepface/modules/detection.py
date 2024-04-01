from typing import List, Union

import numpy
from tqdm import tqdm

from deepface.core import imgutils
from deepface.core.detector import Detector
from deepface.core.types import BoxDimensions


def detect_faces(
    img: Union[str, numpy.ndarray],
    detector: Union[str, Detector] = Detector.get_default(),
    min_dims: BoxDimensions = BoxDimensions(25, 25),
    raise_notfound: bool = False,
) -> Detector.Results:
    """
    Detect faces in an image

    Args:
    -----
        img: image path or numpy array
        detector: detector instance or string
        min_dims: minimum dimensions for detected faces
        raise_notfound: raise exception if no faces are detected

    Returns:
    --------
        detection results

    Raises:
    -------
        ValueError: if the input is invalid
        Any other exceptions raised by the detector or image loading
        functions
    """

    detector_instance = Detector.instance(detector)
    img, tag = imgutils.load_image(img)
    results: Detector.Results = detector_instance.process(
        img,
        tag=tag,
        min_dims=min_dims,
        raise_notfound=raise_notfound,
    )
    return results


def batch_detect_faces(
    imgs: Union[List[str], numpy.ndarray],
    detector: Union[str, Detector] = Detector.get_default(),
    min_dims: BoxDimensions = BoxDimensions(25, 25),
) -> List[Detector.Results]:
    """
    Detect faces in a batch of images   

    Args:
    -----
        imgs: list of image paths or numpy arrays
        detector: detector instance or string
        min_dims: minimum dimensions for detected faces

    Returns:
    --------
        list of detection results

    Raises:
    -------
        ValueError: if the input is invalid
        Any other exceptions raised by the detector or image loading
        functions
    """
    detector_instance = Detector.instance(detector)
    results: List[Detector.Results] = []

    if isinstance(imgs, numpy.ndarray):
        if not imgs.ndim == 4:
            raise ValueError("Expected 4D array for batch processing")
        for i in tqdm(range(imgs.shape[0]), ascii=True, desc="Batch detecting"):
            results.append(
                detect_faces(imgs[i], detector=detector_instance, min_dims=min_dims)
            )

    if isinstance(imgs, list):
        if not all(isinstance(img, str) for img in imgs):
            raise ValueError("Expected list of strings for batch processing")
        for item in tqdm(imgs, ascii=True, desc="Batch detecting"):
            results.append(
                detect_faces(item, detector=detector_instance, min_dims=min_dims)
            )

    return results
