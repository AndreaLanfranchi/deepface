import os
import time
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import cv2
from cv2.typing import MatLike

import pandas as pd

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def analysis(
    db_path:str,
    model_name:str ="VGG-Face",
    detector_backend:str="opencv",
    distance_metric:str="cosine",
    enable_face_analysis:bool=True,
    source:int=0,
    time_threshold: int =3,
    frame_threshold: int =5,
    silent: bool = False,
):

    # Parameter validation
    time_threshold = max(1, min(time_threshold, 10)) # In range [1, 10]
    frame_threshold = max(1, min(frame_threshold, 5)) # In range [1, 5]

    # Constants
    capture_window_title: str = "Capture"
    text_color = (255, 255, 255)
    enable_emotion = True
    enable_age_gender = True

    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    model: FacialRecognition = DeepFace.build_model(model_name=model_name)

    # find custom values for this input set
    target_size = model.input_shape

    if enable_face_analysis:
        DeepFace.build_model(model_name="Race")
        if enable_age_gender:
            DeepFace.build_model(model_name="Age")
            DeepFace.build_model(model_name="Gender")
        if enable_emotion:
            DeepFace.build_model(model_name="Emotion")

    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros(target_size),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False, # This fake imae does not contain a face
        silent=silent,
    )
    # -----------------------
    # visualization

    tic = time.time()
    logger.info("Starting capture source ...")
    cap = cv2.VideoCapture(source)  # webcam
    elapsed = time.time() - tic
    logger.info(f"Started capture source in {elapsed:.5f} seconds")

    # Logic descritpion:
    # --------------------------------
    # 1. Continue read from capture source until at least a defined
    #    number of captured frames contains a detected face. The number of
    #    frames is defined by frame_threshold. The sequence of frames
    #    must be continuous: if a frame does not contain a face, the
    #    counter is reset.
    # 2. Once the sequence of frames is detected, the frame with the
    #    largest facial area is selected as the best capture.
    #    Consider more than one face can be detected so we sum
    #    the areas of all detected faces in a frame
    # 3. The best capture eventually undergoes facial recognition
    #    which will produce the stiching of the matching face(s)
    #    to the boxed detected face(s) in the best capture.
    # 4. The best capture (altered) is then displayed for a few
    #    seconds with a countodown box till time_threshold is reached.
    #
    # The whole loop gracefully exits when the user presses 'q' or
    # the capture source is closed/failed.


    # For each good capture, store the image and the analysis result
    good_captures : List[Tuple[MatLike, List[Dict[str, Any]]]] = []

    while True:
        try:
            capture_successful, captured_frame = cap.read()
            if not capture_successful:
                raise IOError("Capture from source failed")

            cv2.imshow(capture_window_title, captured_frame)
            __cv2_refresh()

            __process_frame(
                frame=captured_frame,
                target_size=target_size,
                detector_backend=detector_backend,
                good_captures=good_captures,
                display_window_title=capture_window_title,
            )

            if len(good_captures) < frame_threshold:
                continue

            # This also resets the good_captures list
            best_capture, best_faces = __get_best_capture(good_captures)
            cv2.imshow(capture_window_title, __box_faces(best_capture, best_faces))
            __cv2_refresh()

            # Perform facial attribute analysis and face recognition
            should_freeze = False
            for item in best_faces:

                # Detect matches for this face
                matching_results = __get_face_matches(
                    best_capture,
                    item["facial_area"],
                    db_path,
                    model_name,
                    distance_metric,
                    silent,
                )
                if len(matching_results) > 0:
                    # Applies matches to the best capture
                    if __process_matches(
                        best_capture,
                        item["facial_area"],
                        matching_results,
                        target_size,
                        detector_backend,
                    ) == True:
                        should_freeze = True
                        __cv2_refresh()

            if should_freeze:
                # Count back from time_threshold to 0
                # Display the best capture for a few seconds
                # with a counting box
                for i in range(time_threshold, 0, -1):
                    cv2.rectangle(best_capture, (10, 10), (70, 50), (67, 67, 67), -10)
                    cv2.putText(
                        best_capture,
                        str(i),
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        text_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(capture_window_title, best_capture)
                    __cv2_refresh(1000)

        except KeyboardInterrupt as ex:
            logger.info(f"{ex.args[0]}")
            break
        except IOError as ex:
            logger.error(f"{ex.args[0]}")
            break
        except Exception as ex:
            logger.error(f"Unexpected error: {ex.args[0]}")
            traceback.print_exc()
            break

    # Clean up
    good_captures.clear()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - tic
    logger.info(f"Total duration {elapsed:.5f} seconds")


# The only way for OpenCV to refresh displayed window(s)
# is to call cv2.waitKey() function. This function waits
# for a given amount of time for a key to be pressed and
# returns the ASCII code of the key pressed.
# We use this as a trick to also check if the 'q' key
# has been pressed by the user.
def __cv2_refresh(timeout:int = 1):
    timeout = max(1, timeout)
    result: int = cv2.waitKey(timeout) & 0xFF
    if result == ord("q"):
        raise KeyboardInterrupt("User requested to stop")

# Process the captured frame as follows:
# - Adds it to the list of good captures when this
#   frame contains a detected face ...
# - ... or resets the list of good captures when no
#   face is detected
# Note ! Faces validly detected but too small are
# discarded. A face is considered too small when its
# area is less than 1/2rd of target_size.
def __process_frame(
        frame: MatLike,
        target_size: Tuple[int, int],
        detector_backend: str,
        good_captures: List[Tuple[MatLike, List[Dict[str, Any]]]],
        display_window_title: str,
):
    try:
        extracted_faces = DeepFace.extract_faces(
            img_path=frame,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=True, # Must find a face or raise an error
            align=False, # Do not align the detected face
        )

        # Remove too small detected faces
        for i in range(len(extracted_faces)-1, -1, -1):
            item = extracted_faces[i]
            w: int = item["facial_area"]["w"]
            h: int = item["facial_area"]["h"]
            if w * h < (target_size[0] * target_size[1]) / 2:
                extracted_faces.pop(i)

        # Treat no-results as error
        if len(extracted_faces) == 0:
            raise ValueError("No face found")

        # Store the good capture and its detection result
        good_captures.append((frame.copy(), extracted_faces))

        # Draw boxes around the detected faces
        cv2.imshow(display_window_title, __box_faces(frame, extracted_faces))
        __cv2_refresh()

    # We only catch the ValueError exception here to reset the good_captures
    # list. Other exceptions are not caught here and will be raised to the
    # caller.
    except ValueError:
        good_captures.clear()

# Get the best capture from the list of good captures
# The best capture is the one with the largest facial
# area. The list of good captures is then reset.
def __get_best_capture(
        good_captures: List[Tuple[MatLike, List[Dict[str, Any]]]]
        ) -> Tuple[MatLike,List[Dict[str, Any]]]:
    best_area:int = 0
    best_index:int = 0
    for i, (_, faces) in enumerate(good_captures):
        area:int = 0
        for face in faces:
            h: int = face["facial_area"]["h"]
            w: int = face["facial_area"]["w"]
            area += h * w
        if area > best_area:
            best_area = area
            best_index = i
    ret = good_captures[best_index]
    good_captures.clear()
    return ret

# Draw box(es) around the detected face(s)
def __box_faces(
        picture: MatLike,
        faces: List[Dict[str, Any]],
        number: Union[int, None] = None
        ) -> MatLike:
    for face in faces:
        x: int = face["facial_area"]["x"]
        y: int = face["facial_area"]["y"]
        w: int = face["facial_area"]["w"]
        h: int = face["facial_area"]["h"]
        picture = __box_face(picture, (x, y, w, h))
        if number is not None:
            cv2.putText(
                picture,
                str(number),
                (int(x + w / 4), int(y + h / 1.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return picture

# Draws a box around the provided region
def __box_face(
    picture: MatLike,
    region: Tuple[int, int, int, int] # (x, y, w, h)
    ) -> MatLike:
    x: int = region[0]
    y: int = region[1]
    w: int = region[2]
    h: int = region[3]
    cv2.rectangle(picture, (x, y), (x + w, y + h), (67, 67, 67), 1)
    return picture

# Crop the face from the picture
def __crop_face(
    picture: MatLike,
    region: Dict[str, int]
    ) -> MatLike:
    x: int = region["x"]
    y: int = region["y"]
    w: int = region["w"]
    h: int = region["h"]
    return picture[y : y + h, x : x + w]

# Get the matches from the dataset
# which are the closest to the detected face
def __get_face_matches(
    picture: MatLike,
    facial_area: Dict[str, Any],
    db_path: str,
    model_name: str,
    distance_metric: str,
    silent: bool,
) -> List[pd.DataFrame]:

    try:
        cropped_face = __crop_face(picture, facial_area)
        matching_results = DeepFace.find(
            img_path=cropped_face,
            db_path=db_path,
            model_name=model_name,
            detector_backend="skip", # Skip detection, we already have the face
            distance_metric=distance_metric,
            enforce_detection=True,
            silent=silent,
        )
        return matching_results
    except ValueError:
        return []

# Process the matches and stick the matching face
# (if any)
def __process_matches(
        picture: MatLike,
        facial_area: Dict[str, Any],
        matching_results: List[pd.DataFrame],
        target_size: Tuple[int, int],
        detector_backend: str,
) -> bool:

    matching_result = matching_results[0]
    if matching_result.shape[0] == 0:
        return False

    matching_item = matching_result.iloc[0]
    matching_identity = matching_item["identity"]

    # TODO This is time consuming
    # Wouldn't be possible to detect the matching
    # from the data returned by the matching_results ?
    try:
        matching_faces = DeepFace.extract_faces(
            img_path=matching_identity,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=False,
        )
    except ValueError:
        return False

    x: int = facial_area["x"]
    y: int = facial_area["y"]
    w: int = facial_area["w"]
    h: int = facial_area["h"]

    matching_face = matching_faces[0]["face"]
    matching_face = cv2.resize(matching_face, (int(h/2.5), int(w/2.5)))
    matching_face *= 255
    matching_face = matching_face[:, :, ::-1]

    # Stick the matching face to the original picture
    # in the lower-right corner of the boxed face
    picture[
        y + h - matching_face.shape[0] : y + h,
        x + w - matching_face.shape[1] : x + w] = matching_face

    return True
