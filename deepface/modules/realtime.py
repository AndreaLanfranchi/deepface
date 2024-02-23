import os
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import cv2
from cv2.typing import MatLike, Point

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks


def analysis(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
    silent: bool = False,
):
    # global variables
    capture_window_title = "Capture"
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

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
        DeepFace.build_model(model_name="Age")
        DeepFace.build_model(model_name="Gender")
        DeepFace.build_model(model_name="Emotion")
    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
        silent=silent,
    )
    # -----------------------
    # visualization
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
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
    # 2. Once the sequence of frames is detected, freeze the last
    #    frame for a number of seconds defined by time_threshold.
    #    During this time, perform facial attribute analysis and
    #    face recognition from matching picture in dataset. The
    #    matching picture and the facial attribute analysis are
    #    displayed in the frozen frame.
    # 3. After the time_threshold, the process restarts from step 1.
    #
    # The whole loop gracefully exits when the user presses 'q' or
    # the capture source is closed/failed.


    # For each good capture, store the image and the analysis result
    good_captures : List[Tuple[MatLike, List[Dict[str, Any]]]] = []
    should_stop: bool = False

    while not should_stop:
        capture_successful, captured_frame = cap.read()
        if not capture_successful:
            break
        cv2.imshow(capture_window_title, captured_frame)
        should_stop = __cv2_trap_q(should_stop)

        # Detection phase 
        try:
            extracted_faces = DeepFace.extract_faces(
                img_path=captured_frame,
                target_size=target_size,
                detector_backend=detector_backend,
                enforce_detection=True, # Must find a face
            )

            # Remove too small detected faces
            for i in range(len(extracted_faces)-1, -1, -1):
                item = extracted_faces[i]
                if item["facial_area"]["w"] <= 130:
                    extracted_faces.pop(i)
            
            # Treat no-results as error
            if not len(extracted_faces) > 0:
                raise ValueError("No face found")

            # Store the good capture and its detection result
            good_captures.append((captured_frame.copy(), extracted_faces))

            # Draw boxes around the detected faces
            cv2.imshow(capture_window_title, __box_faces(captured_frame, extracted_faces, len(good_captures)))
            should_stop = __cv2_trap_q(should_stop)
            
        except ValueError: # No face found
            # No face found, reset what collected so far
            good_captures.clear()
            continue
        except Exception as e:
            raise RuntimeError(f"Error during face detection: {e}") from e

        # Continue until at least a defined number of frames contains
        # validly extracted face(s)
        if len(good_captures) < frame_threshold:
            continue

        # Now that we have enough frames pick the best one
        # (the one with the widest facial area)
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

        best_capture, best_faces = good_captures[best_index]
        cv2.imshow(capture_window_title, __box_faces(best_capture, best_faces))
        should_stop = __cv2_trap_q(should_stop)

        resolution_x:int = best_capture.shape[1]
        resolution_y:int = best_capture.shape[0]

        # Perform facial attribute analysis and face recognition
        for item in best_faces:
            face_image = __crop_face(best_capture, item["facial_area"])
            x: int = face["facial_area"]["x"]
            y: int = face["facial_area"]["y"]
            w: int = face["facial_area"]["w"]
            h: int = face["facial_area"]["h"]


            # TODO Facial attribute analysis

            # Face recognition
            matching_results = DeepFace.find(
                img_path=face_image,
                db_path=db_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=silent,
            )
            if len(matching_results) == 0:
                logger.info("No matches found")
                continue
            matching_result = matching_results[0]
            if matching_result.shape[0] == 0:
                logger.info("No matches found")
                continue

            matching_item = matching_result.iloc[0]
            matching_identity = matching_item["identity"]
            extracted_faces = DeepFace.extract_faces(
                img_path=matching_identity,
                target_size=(pivot_img_size,pivot_img_size),
                detector_backend=detector_backend,
                enforce_detection=False,
                align=False,
            )

            if len(extracted_faces) == 0:
                logger.info("No face found in matching picture")
                continue

            pivot_img = extracted_faces[0]["face"]
            pivot_img *= 255
            pivot_img = pivot_img[:, :, ::-1]

            # Draw the matching picture and the facial attribute analysis
            # in the frozen frame
            best_capture[
                y - pivot_img_size : y ,
                x + w : x + w + pivot_img_size] = pivot_img
            cv2.imshow(capture_window_title, best_capture)
            should_stop = __cv2_trap_q(should_stop)

            


        time.sleep(2)

        # Reset the good captures
        good_captures.clear()



    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()

# Actually this also serves the purpose to refresh
# the capture window and check for the 'q' key to be pressed
def __cv2_trap_q(trapped:bool) -> bool:
    val = cv2.waitKey(1) & 0xFF
    if(trapped):
        return trapped
    return val == ord("q")

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
        cv2.rectangle(picture, (x, y), (x + w, y + h), (67, 67, 67), 1)
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