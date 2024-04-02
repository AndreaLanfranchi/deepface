from typing import List, Optional, Union

import os
import numpy
from tqdm import tqdm

from deepface.core import imgutils
from deepface.core.detector import Detector
from deepface.core.types import BoxDimensions


def detect_faces(
    img: Union[str, numpy.ndarray],
    tag: Optional[str] = None,
    detector: Optional[Union[str, Detector]] = None,
    min_dims: Optional[BoxDimensions] = None,
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
    returned_img, returned_tag = imgutils.load_image(img)
    if tag is not None:
        if not isinstance(tag, str):
            tag = str(tag).strip()
        returned_tag = f"{tag} ({returned_tag})"
    results: Detector.Results = detector_instance.process(
        img=returned_img,
        tag=returned_tag,
        min_dims=min_dims,
        raise_notfound=raise_notfound,
    )
    return results


def batch_detect_faces(
    imgs: Union[str, List[str], numpy.ndarray],
    detector: Optional[Union[str, Detector]] = None,
    min_dims: Optional[BoxDimensions] = None,
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
            # TODO: if the following raises decide whether the skip the
            # offending image or let the exception to pop up
            results.append(
                detect_faces(
                    imgs[i],
                    tag=str(i),
                    detector=detector_instance,
                    min_dims=min_dims,
                )
            )

    if isinstance(imgs, str):
        if os.path.isfile(imgs):
            imgs = [imgs,]
        elif os.path.isdir(imgs):
            file_list = imgutils.get_all_valid_files(imgs, recurse=True)
            imgs = file_list

    if isinstance(imgs, list):
        if len(imgs) == 0:
            raise ValueError("Empty list of images for batch processing")
        for item in tqdm(imgs, ascii=True, desc="Batch detecting"):
            if not isinstance(item, str):
                continue
            # TODO: if the following raises decide whether the skip the
            # offending image or let the exception to pop up
            file_name:str = item.strip()
            if not imgutils.is_valid_image_file(filename=file_name):
                continue
            results.append(
                detect_faces(item, detector=detector_instance, min_dims=min_dims)
            )

    return results
