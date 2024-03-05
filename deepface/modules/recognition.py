# built-in dependencies
import os
import re
import pickle
from typing import List, Set, Union, Optional
import time

# 3rd party dependencies
import numpy
import pandas
from tqdm import tqdm

# project dependencies
from deepface.commons.logger import Logger
from deepface.modules import representation, detection, modeling, verification
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger(module="deepface/modules/recognition.py")


def find(
    img_path: Union[str, numpy.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base"
) -> List[pandas.DataFrame]:
    """
    Identify individuals in a database

    Args:
        img_path (str or numpy.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        db_path (string): Path to the folder containing image files. Must be a directory.
            All valid pictures (jpg, jpeg, png) image files with validly detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

    Returns:
        results (List[pandas.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
            to the identity information for an individual detected in the source image.
            The DataFrame columns include:

            - 'identity': Identity label of the detected individual.

            - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                    target face in the database.

            - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                    detected face in the source image.

            - 'threshold': threshold to determine a pair whether same person or different persons

            - 'distance': Similarity score between the faces based on the
                    specified model and distance metric
    """

    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError(f"{db_path} does not exist or is not a directory.")

    model: FacialRecognition = modeling.get_recognition_model(model_name)
    target_size = model.input_shape

    # ---------------------------------------

    file_name = f"representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()
    datastore_path = os.path.join(db_path, file_name)
    representations = []

    # This is the "record template" for each item in the pickle file
    template_cols: List[str] = [
        "identity",
        f"{model_name}_representation",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    # Ensure the proper pickle file exists
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as f:
            pickle.dump([], f)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    # Check if the representations are out-of-date
    if len(representations) > 0:
        if len(representations[0]) != len(template_cols):
            raise ValueError(
                f"Seems existing {datastore_path} is out-of-the-date."
                "Please delete it and re-run."
            )

    # Get the list of images on storage
    storage_images:Set[str] = _list_image_files(path=db_path)
    pickled_images:Set[str] = {representation[0] for representation in representations}

    # Enforce data consistency amongst on disk images and pickle file
    must_save_pickle = False
    new_images:Set[str] = storage_images - pickled_images # images added to storage
    old_images:Set[str] = pickled_images - storage_images # images removed from storage

    if len(new_images) > 0 or len(old_images) > 0:
        logger.info(f"Found {len(new_images)} new images and {len(old_images)} removed images")

    # remove old images first
    if len(old_images)>0:
        representations = [rep for rep in representations if rep[0] not in old_images]
        must_save_pickle = True

    # find representations for new images
    if len(new_images)>0:
        representations += _find_bulk_embeddings(
            employees=new_images,
            model_name=model_name,
            target_size=target_size,
            detector_backend=detector_backend,
            align=align,
            normalization=normalization
        ) # add new images
        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)
            logger.info(f"There are now {len(representations)} representations in {file_name}")

    # Should we have no representations bailout
    if len(representations) == 0:
        logger.debug(f"find function duration {(time.time() - tic):0.5f} seconds")
        return []

    # ----------------------------
    # now, we got representations for facial database
    df = pandas.DataFrame(
        representations,
        columns=template_cols,
    )

    try:
        # img path might have more than once face
        source_objs = detection.detect_faces(
            source=img_path,
            target_size=target_size,
            detector=detector_backend,
            grayscale=False,
            align=align,
            expand_percentage=expand_percentage,
        )
    except ValueError:
        source_objs = []

    resp_obj = []

    distance_metric = distance_metric.lower().strip()
    if distance_metric == "cosine":
        distance_fn = verification.find_cosine_distance
    elif distance_metric == "euclidean":
        distance_fn = verification.find_euclidean_distance
    elif distance_metric == "euclidean_l2":
        distance_fn = verification.find_euclidean_l2_distance
    else:
        raise NotImplementedError("Invalid distance_metric passed : ", distance_metric)


    for source_obj in source_objs:
        source_img = source_obj["face"]
        source_region = source_obj["facial_area"]
        target_embedding_obj = representation.represent(
            img_path=source_img,
            model_name=model_name,
            detector_backend="donotdetect",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = source_region["x"]
        result_df["source_y"] = source_region["y"]
        result_df["source_w"] = source_region["w"]
        result_df["source_h"] = source_region["h"]

        distances = []
        for _, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]
            if source_representation is None:
                distances.append(float("inf")) # no representation for this image
                continue

            target_dims = len(list(target_representation))
            source_dims = len(list(source_representation))
            if target_dims != source_dims:
                raise ValueError(
                    "Representation dimensions mismatch!\n" + 
                    f"Target : {target_dims} vs Source: {source_dims}.\n" +
                    f"Please delete {file_name} and re-run."
                )

            distances.append(distance_fn(source_representation, target_representation))


        target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

        result_df["threshold"] = target_threshold
        result_df["distance"] = distances

        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df["distance"] <= target_threshold]
        result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

        resp_obj.append(result_df)


    logger.debug(f"find function duration {(time.time() - tic):0.5f} seconds")
    return resp_obj


_image_file_pattern = re.compile(r".*\.(jpg|jpeg|png)$", re.IGNORECASE) # Mimic static variable

def _list_image_files(path: str) -> Set[str]:
    """
    List images in a given path
    Args:
        path (str): Directory location where to get the collection
            of image file names
    Returns:
        images (set): Unique list of image file names combined with
            the path
    Raises:
        IOError: if the path does not exist or is not a directory
    """
    results: Set[str] = set()
    if not os.path.isdir(path):
        raise IOError(f"Path {path} does not exist or is not a directory!")

    for file_name in os.listdir(path):
        if _image_file_pattern.match(file_name):
            results.add(os.path.join(path, file_name))

    return results

def _find_bulk_embeddings(
    employees: List[str],
    model_name: str = "VGG-Face",
    target_size: tuple = (224, 224),
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base"
):
    """
    Find embeddings of a list of images

    Args:
        employees (list): list of exact image paths

        model_name (str): facial recognition model name

        target_size (tuple): expected input shape of facial recognition model

        detector_backend (str): face detector model name

        align (bool): enable or disable alignment of image
            before feeding to facial recognition model

        expand_percentage (int): expand detected facial area with a
            percentage (default is 0).

        normalization (bool): normalization technique

    Returns:
        representations (list): pivot list of embeddings with
            image name and detected face area's coordinates
    """
    representations = []
    for employee in tqdm(
        employees,
        desc="Finding representations"
    ):
        try:
            img_objs = detection.detect_faces(
                source=employee,
                target_size=target_size,
                detector=detector_backend,
                grayscale=False,
                align=align,
                expand_percentage=expand_percentage,
            )
        except ValueError as err:
            logger.error(
                f"Exception while extracting faces from {employee}: {str(err)}"
            )
            img_objs = []

        if len(img_objs) == 0:
            logger.warn(f"No face detected in {employee}. It will be skipped in detection.")
            representations.append((employee, None, 0, 0, 0, 0))
        else:
            for img_obj in img_objs:
                img_content = img_obj["face"]
                img_region = img_obj["facial_area"]
                embedding_obj = representation.represent(
                    img_path=img_content,
                    model_name=model_name,
                    detector_backend="donotdetect",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]
                representations.append((
                    employee,
                    img_representation,
                    img_region["x"],
                    img_region["y"],
                    img_region["w"],
                    img_region["h"]
                    ))

    return representations
