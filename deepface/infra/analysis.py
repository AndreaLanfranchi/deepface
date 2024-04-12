from typing import Dict, List, Optional, Union

import os
import numpy
from tqdm import tqdm

from deepface.core import imgutils
from deepface.core.detector import Detector
from deepface.core.analyzer import Analyzer
from deepface.core.types import BoxDimensions, DetectedFace


def analyze_faces(
    inp: Union[str, numpy.ndarray],
    tag: Optional[str] = None,
    detector: Union[str, Detector] = "default",
    attributes: Optional[Union[str, List[str]]] = None,
    min_dims: Optional[BoxDimensions] = None,
    min_confidence: Optional[float] = None,
    key_points: bool = False,
    raise_notfound: bool = False,
) -> Dict[str, Optional[List[DetectedFace]]]:
    """
    Analyzes faces attributes in an image

    Args:
    -----
        `input`: image path or numpy array

        `tag`: tag for the image

        `detector`: detector instance or string. The model to use for face
        detection. If None, the default detector is assumed

        `attributes`: a list of attributes to analyze. If None, all available
        attributes are analyzed

        `min_dims`: minimum dimensions for detected faces. Detected faces
        with bounding boxes smaller than this are ignored. If None, the
        no filtering is applied

        `min_confidence`: minimum confidence for detected faces. If None, the
        default confidence typical for the detector is assumed.

        `key_points`: Whether to detect key points along with the face

        `raise_notfound`: Whether to raise an exception if no faces are detected

    """

    if attributes is None:
        attributes = Analyzer.get_available_attributes()
    if isinstance(attributes, str):
        attributes = [
            attributes,
        ]
    if not isinstance(attributes, list):
        what: str = (
            "Invalid `attributes` argument type. Expected [None | str | List[str]]"
        )
        what += f" but got {type(attributes)}"
        raise TypeError(what)

    analyzer_instances: List[Analyzer] = [
        Analyzer.instance(attr) for attr in attributes
    ]

    results: Dict[str, Optional[List[DetectedFace]]] = {}
    img, tag = imgutils.load_image(inp, tag)
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
        for detection in detector_results.detections:
            inner_results: Dict[str, str] = {}
            for analyzer_instance in analyzer_instances:
                analisis_result = analyzer_instance.process(
                    detector_results.img,
                    detection.bounding_box,
                )
                inner_results[analyzer_instance.name] = analisis_result.value
            detection.set_attributes(inner_results)
        results[tag] = detector_results.detections

    return results


def batch_analyze_faces(
    inputs: Union[str, List[str], numpy.ndarray],
    detector: Union[str, Detector] = "default",
    attributes: Optional[Union[str, List[str]]] = None,
    min_dims: Optional[BoxDimensions] = None,
    min_confidence: Optional[float] = None,
    key_points: bool = False,
    raise_notfound: bool = False,
    recurse: bool = True,
) -> Dict[str, Optional[List[DetectedFace]]]:
    """
    Analyzes faces attributes from a batch of images

    Args:
    -----
        `inputs`: list of image paths or numpy arrays\n
        In case of numpy arrays, the input should be a 4D array\n
        In case of strings the behavior is the following:
        - if the string is a file, it is considered an image file
        - if the string is a directory, all valid image files in the
          directory are considered

        `detector`: detector instance or string. The model to use for face
        detection. If None, the default detector is assumed

        `attributes`: a list of attributes to analyze. If None, all available
        attributes are analyzed

        `min_dims`: minimum dimensions for detected faces. Detected faces
        with bounding boxes smaller than this are ignored. If None, the
        no filtering is applied

        `min_confidence`: minimum confidence for detected faces. If None, the
        default confidence typical for the detector is assumed.

        `key_points`: Whether to detect key points along with the face

        `raise_notfound`: Whether to raise an exception if no faces are detected

    """

    if inputs is None:
        raise ValueError("Argument [inputs] cannot be None")

    detector_instance = Detector.instance(detector)
    results: Dict[str, Optional[List[DetectedFace]]] = {}

    if isinstance(inputs, numpy.ndarray):
        if not inputs.ndim == 4:
            raise ValueError("Expected 4D array for batch processing")
        for i in tqdm(range(inputs.shape[0]), ascii=True, desc="Batch analyzing"):
            # TODO: if the following raises decide whether the skip the
            # offending image or let the exception to pop up
            item_results = analyze_faces(
                inputs[i],
                tag=f"{i}/{inputs.shape[0]}",
                detector=detector_instance,
                attributes=attributes,
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
        for file in tqdm(files, ascii=True, desc="Batch analyzing"):
            item_results = analyze_faces(
                file,
                tag=file,
                detector=detector_instance,
                attributes=attributes,
                min_dims=min_dims,
                min_confidence=min_confidence,
                key_points=key_points,
                raise_notfound=raise_notfound,
            )
            results.update(item_results)

    elif raise_notfound:
        raise FileNotFoundError("No valid image files found in the input")

    return results
