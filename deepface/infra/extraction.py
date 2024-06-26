from typing import Dict, List, Optional, Union

import os
import numpy
from tqdm import tqdm

from deepface.core import imgutils
from deepface.core.detector import Detector
from deepface.core.extractor import Extractor
from deepface.core.types import BoxDimensions, DetectedFace


def extract_faces(
    inp: Union[str, numpy.ndarray],
    tag: Optional[str] = None,
    detector: Union[str, Detector] = "default",
    extractor: Union[str, Extractor] = "default",
    min_dims: Optional[BoxDimensions] = None,
    min_confidence: Optional[float] = None,
    key_points: bool = False,
    raise_notfound: bool = False,
) -> Dict[str, Optional[List[DetectedFace]]]:
    """
    Extract faces from an image

    Args:
    -----
        `input`: image path or numpy array

        `tag`: tag for the image

        `detector`: detector instance or string. The model to use for face
        detection. The literal "default" is used to indicate the default

        `extractor`: extractor instance or string. The model to use for face
        extraction. The literal "default" is used to indicate the default

        `min_dims`: minimum dimensions for detected faces. Detected faces
        with bounding boxes smaller than this are ignored. If None, the
        no filtering is applied

        `min_confidence`: minimum confidence for detected faces. If None, the
        default confidence typical for the detector is assumed.

        `key_points`: Whether to detect key points along with the face

        `raise_notfound`: Whether to raise an exception if no faces are detected

    Returns:
    --------
        A list of tuples containing the tag, the detected face and the
        extracted face representation

    Raises:
    -------
        ValueError: if the input is invalid
        Any other exceptions raised by the detector or image loading
        functions
    """

    results: Dict[str, Optional[List[DetectedFace]]] = {}
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

    tag = "<unknown>" if tag is None else tag
    if not detector_results:
        results[tag] = None
    else:
        extractor_instance = Extractor.instance(extractor)
        for detection in detector_results.detections:
            extraction_result = extractor_instance.process(img, detection.bounding_box)
            detection.set_embeddings(extraction_result)

        results[tag] = detector_results.detections

    return results


def batch_extract_faces(
    inputs: Union[str, List[str], numpy.ndarray],
    detector: Union[str, Detector] = "default",
    extractor: Union[str, Extractor] = "default",
    min_dims: Optional[BoxDimensions] = None,
    min_confidence: Optional[float] = None,
    key_points: bool = False,
    raise_notfound: bool = False,
    recurse: bool = True,
) -> Dict[str, Optional[List[DetectedFace]]]:
    """
    Extract faces from a batch of images

    Args:
    -----
        `inputs`: list of image paths or numpy arrays\n
        In case of numpy arrays, the input should be a 4D array\n
        In case of strings the behavior is the following:
        - if the string is a file, it is considered an image file
        - if the string is a directory, all valid image files in the
          directory are considered

        `detector`: detector instance or string. The model to use for face
        detection. The literal "default" is used to indicate the default

        `extractor`: extractor instance or string. The model to use for face
        extraction. The literal "default" is used to indicate the default

        `min_dims`: minimum dimensions for detected faces. Detected faces
        with bounding boxes smaller than this are ignored. If None, the
        no filtering is applied

        `min_confidence`: minimum confidence for detected faces. If None, the
        default confidence typical for the detector is assumed.

        `key_points`: Whether to detect key points along with the face

        `raise_notfound`: Whether to raise an exception if no faces are detected

        `recurse`: if the input is a directory, recurse into subdirectories

    Returns:
    --------
        A list of tuples containing the tag, the detected face and the
        extracted face representation

    Raises:
    -------
        ValueError: if the input is invalid
        Any other exceptions raised by the detector or image loading
        functions
    """

    if inputs is None:
        raise ValueError("Argument [inputs] cannot be None")

    detector_instance = Detector.instance(detector)
    extractor_instance = Extractor.instance(extractor)
    results: Dict[str, Optional[List[DetectedFace]]] = {}

    if isinstance(inputs, numpy.ndarray):
        if not inputs.ndim == 4:
            raise ValueError("Expected 4D array for batch processing")
        for i in tqdm(range(inputs.shape[0]), ascii=True, desc="Batch detecting"):
            # TODO: if the following raises decide whether the skip the
            # offending image or let the exception to pop up
            item_results = extract_faces(
                inputs[i],
                tag=f"{i}/{inputs.shape[0]}",
                detector=detector_instance,
                extractor=extractor_instance,
                min_dims=min_dims,
                min_confidence=min_confidence,
                key_points=key_points,
                raise_notfound=raise_notfound,
            )
            results.update(item_results)

        return results

    sources: List[str] = []
    if not isinstance(inputs, list):
        sources = [
            str(inputs),
        ]
    else:
        sources = inputs

    if 0 == len(sources):
        raise ValueError("Empty list of images for batch processing")

    files: List[str] = []
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
        for file in tqdm(files, ascii=True, desc="Batch extracting"):
            item_results = extract_faces(
                file,
                tag=file,
                detector=detector_instance,
                extractor=extractor_instance,
                min_dims=min_dims,
                min_confidence=min_confidence,
                key_points=key_points,
                raise_notfound=raise_notfound,
            )
            results.update(item_results)

    elif raise_notfound:
        raise FileNotFoundError("No valid image files found in the input")

    return results
