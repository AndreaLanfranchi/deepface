from typing import List, Optional, Union

import os
import numpy
from tqdm import tqdm

from deepface.core import imgutils
from deepface.core.detector import Detector
from deepface.core.types import BoxDimensions


def detect_faces(
    inp: Union[str, numpy.ndarray],
    tag: Optional[str] = None,
    detector: Optional[Union[str, Detector]] = None,
    min_confidence: Optional[float] = None,
    min_dims: Optional[BoxDimensions] = None,
    key_points: bool = True,
    raise_notfound: bool = False,
) -> Detector.Results:
    """
    Detect faces in an image

    Args:
    -----
        `input`: image path or numpy array

        `detector`: detector instance or name. If None, the default detector
        is assumed

        `min_confidence`: minimum confidence for detected faces. If None, the
        default confidence typical for the detector is assumed

        `min_dims`: minimum dimensions for detected faces

        `key_points`: whether to detect key points

        `raise_notfound`: if True, raise an exception if no faces are found

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
    img, tag = imgutils.load_image(inp, tag=tag)
    results: Detector.Results = detector_instance.process(
        img=img,
        tag=tag,
        min_confidence=min_confidence,
        min_dims=min_dims,
        key_points=key_points,
        raise_notfound=raise_notfound,
    )
    return results


def batch_detect_faces(
    inputs: Union[str, List[str], numpy.ndarray],
    detector: Optional[Union[str, Detector]] = None,
    min_confidence: Optional[float] = None,
    min_dims: Optional[BoxDimensions] = None,
    key_points: bool = True,
    raise_notfound: bool = False,
    recurse: bool = True,
) -> List[Detector.Results]:
    """
    Detect faces in a batch of images

    Args:
    -----
        `inputs`: list of image paths or numpy arrays\n
        In case of numpy arrays, the input should be a 4D array\n
        In case of strings the behavior is the following:
        - if the string is a file, it is considered an image file
        - if the string is a directory, all valid image files in the
          directory are considered
      
        `detector`: detector instance or name. If None, the default detector
        is assumed

        `min_confidence`: minimum confidence for detected faces. If None, the
        default confidence typical for the detector is assumed

        `min_dims`: minimum dimensions for detected faces

        `key_points`: whether to detect key points

        `raise_notfound`: if True, raise an exception if no faces are found

        `recurse`: if the input is a directory, recurse into subdirectories

    Returns:
    --------
        A list of `Detector.Results` class instances

    Raises:
    -------
        `ValueError`: if the input is invalid

        Any other exception raised by the detector or image loading
        functions
    """

    if inputs is None:
        raise ValueError("Argument [inputs] cannot be None")

    detector_instance = Detector.instance(detector)
    results: List[Detector.Results] = []

    if isinstance(inputs, numpy.ndarray):
        if not inputs.ndim == 4:
            raise ValueError("Expected 4D array for batch processing")
        for i in tqdm(range(inputs.shape[0]), ascii=True, desc="Batch detecting"):
            # TODO: if the following raises decide whether the skip the
            # offending image or let the exception to pop up
            results.append(
                detect_faces(
                    inputs[i],
                    tag= f"{i}/{inputs.shape[0]}",
                    detector=detector_instance,
                    min_confidence=min_confidence,
                    min_dims=min_dims,
                    key_points=key_points,
                    raise_notfound=raise_notfound,
                )
            )

        return results

    sources:List[str] = []
    if not isinstance(inputs, list):
        sources = [str(inputs),]
    else:
        sources = inputs

    if len(sources) == 0:
        raise ValueError("Empty list of images for batch processing")

    files:List[str] = []
    for source in sources:
        if not isinstance(source, str):
            continue
        if os.path.isfile(source):
            files.append(source)
        elif os.path.isdir(source):
            file_list = imgutils.get_all_image_files(
                source,
                recurse=recurse,
                check_ext=True,
            )
            files.extend(file_list)

    if 0 != len(files):
        for file in tqdm(files, ascii=True, desc="Batch detecting"):
            file_name:str = file.strip()
            results.append(
                detect_faces(
                    inp=file_name,
                    tag=file_name,
                    detector=detector_instance,
                    min_confidence=min_confidence,
                    key_points=key_points,
                    min_dims=min_dims,
                )
            )

    return results
