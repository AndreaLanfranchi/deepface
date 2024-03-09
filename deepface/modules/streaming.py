import sys
import os
import time
import threading
import traceback
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy
import cv2
from cv2.typing import MatLike

import pandas

from deepface import DeepFace
from deepface.core.detector import Detector
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger.get_instance()

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class Stream(threading.Thread):
    """
    A class to read frames from a video capture source in a separate thread.
    This is necessary due to the "freeze" feature of this module which
    causes the capture source to accumulate frames in the read queue which
    cannot be processed fast enoungh when the main thread resumes.
    The higher FPS connected cameras can provide the more the problem is
    noticeable.
    Credits for the implementation of this class go to the following:
    https://stackoverflow.com/a/65191619
    """

    def __init__(
        self,
        source: Union[str, int] = int(0),
        buffer_frame_size: int = 3,
    ):
        super().__init__(name="VideoStream")
        self.__vc = cv2.VideoCapture(source)
        self.__vc.set(cv2.CAP_PROP_BUFFERSIZE, buffer_frame_size)
        self.__last_read: bool = True
        self.__last_frame: MatLike = numpy.array([])
        self.__stop: bool = False
        self.start()

    def run(self):
        logger.info("Capture thread started")
        try:
            while not self.__stop:
                self.__last_read, self.__last_frame = self.__vc.read()
                if not self.__last_read:
                    self.__vc.release()
                    self.__stop = True
        except:
            # Any exception will complete the task
            self.__stop = True

    def read(self) -> Tuple[bool, MatLike]:
        if self.__stop:
            raise IOError("Capture source failed")
        return (self.__last_read, self.__last_frame)

    def stop(self):
        self.__stop = True


def analysis(
    db_path: str,
    model_name: str = "VGG-Face",
    detector: Optional[str] = None,
    distance_metric: str = "cosine",
    analyzers: Optional[List[str]] = None,
    source: Union[str, int] = int(0),
    freeze_time_seconds: int = 3,
    valid_frames_count: int = 5,
    faces_count_threshold: int = sys.maxsize,
):

    # Parameter validation
    freeze_time_seconds = max(
        1, min(freeze_time_seconds, 10)
    )  # In range [1, 10] positive
    valid_frames_count = max(1, min(valid_frames_count, 5))  # In range [1, 5] positive
    faces_count_threshold = max(1, faces_count_threshold)  # In range [1, inf] positive

    detector = Detector.instance(detector)

    # Constants
    capture_window_title: str = "Capture"
    text_color = (255, 255, 255)

    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    model: FacialRecognition = DeepFace.get_recognition_model(name=model_name)

    # find custom values for this input set
    target_size = model.input_shape

    # Lazy load the analysis models (if needed)
    if analyzers is not None:
        for i in range(len(analyzers) - 1, -1, -1):
            try:
                _ = DeepFace.get_analysis_model(name=analyzers[i])
            except KeyError as ex:
                logger.error(ex.args[0])
                analyzers.pop(i)

    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    _ = DeepFace.find(
        img_path=numpy.zeros(target_size, dtype=numpy.uint8),
        db_path=db_path,
        model_name=model_name,
        detector=detector,
        distance_metric=distance_metric,
    )
    # -----------------------
    # visualization

    tic = time.time()
    logger.info("Starting capture source ...")
    stream = Stream(source=source)
    time.sleep(1)  # Wait for the stream to start - 1 second should be enough
    logger.info(f"Started capture source in {(time.time() - tic):.5f} seconds")

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
    good_captures: List[Tuple[MatLike, List[Dict[str, Any]]]] = []

    while True:
        try:
            capture_successful, captured_frame = stream.read()
            if not capture_successful:
                raise IOError("Capture from source failed")
            if captured_frame.size == 0:
                continue

            cv2.imshow(capture_window_title, captured_frame)
            __cv2_refresh()

            __process_frame(
                frame=captured_frame,
                target_size=target_size,
                faces_count_threshold=faces_count_threshold,
                detector=detector,
                good_captures=good_captures,
                display_window_title=capture_window_title,
            )

            if len(good_captures) < valid_frames_count:
                continue

            # This also resets the good_captures list
            best_capture, best_faces = __get_best_capture(good_captures)
            cv2.imshow(capture_window_title, __box_faces(best_capture, best_faces))
            __cv2_refresh()

            # TODO : Perform facial attribute analysis and face recognition

            # Display the results
            should_freeze = False
            for item in best_faces:

                # Detect matches for this face
                matching_results = __get_face_matches(
                    face=item["face"],
                    db_path=db_path,
                    model_name=model_name,
                    distance_metric=distance_metric,
                )
                if len(matching_results) > 0:
                    # Applies matches to the best capture
                    if (
                        __process_matches(
                            best_capture,
                            item["facial_area"],
                            matching_results,
                            detector,
                        )
                        == True
                    ):
                        should_freeze = True
                        __cv2_refresh()

            if should_freeze:
                # Count back from time_threshold to 0
                # Display the best capture for a few seconds
                # with a counting box
                for i in range(freeze_time_seconds, 0, -1):
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
    stream.stop()
    stream.join()
    cv2.destroyAllWindows()

    logger.info(f"Total duration {(time.time() - tic):.5f} seconds")


# The only way for OpenCV to refresh displayed window(s)
# is to call cv2.waitKey() function. This function waits
# for a given amount of time for a key to be pressed and
# returns the ASCII code of the key pressed.
# We use this as a trick to also check if the 'q' key
# has been pressed by the user.
def __cv2_refresh(timeout: int = 1):
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
    faces_count_threshold: int,
    good_captures: List[Tuple[MatLike, List[Dict[str, Any]]]],
    display_window_title: str,
    detector: Optional[Union[str, Detector]] = None,
):
    try:
        extracted_faces = DeepFace.detect_faces(
            img_path=frame,
            target_size=target_size,
            detector=detector,
            align=False,  # Do not align the detected face
        )

        # Remove too small detected faces
        # for i in range(len(extracted_faces)-1, -1, -1):
        #     item = extracted_faces[i]
        #     w: int = item["facial_area"]["w"]
        #     h: int = item["facial_area"]["h"]
        #     if w * h < (target_size[0] * target_size[1]) / 2:
        #         extracted_faces.pop(i)

        # Treat no-results or too many results as error
        if len(extracted_faces) == 0:
            raise ValueError("No face found")
        if len(extracted_faces) > faces_count_threshold:
            raise OverflowError("Too many faces found")

        # Store the good capture and its detection result
        good_captures.append((frame.copy(), extracted_faces))

        # Draw boxes around the detected faces
        cv2.imshow(display_window_title, __box_faces(frame, extracted_faces))
        __cv2_refresh()

    # We only catch the ValueError and OverflowError exceptions here to reset
    # the good_captures list. Other exceptions are not caught here and will be
    # raised to the caller.
    except ValueError:
        good_captures.clear()
    except OverflowError:
        good_captures.clear()


# Get the best capture from the list of good captures
# The best capture is the one with the largest facial
# area. The list of good captures is then reset.
def __get_best_capture(
    good_captures: List[Tuple[MatLike, List[Dict[str, Any]]]]
) -> Tuple[MatLike, List[Dict[str, Any]]]:
    best_area: int = 0
    best_index: int = 0
    for i, (_, faces) in enumerate(good_captures):
        area: int = 0
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
    picture: MatLike, faces: List[Dict[str, Any]], number: Union[int, None] = None
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
    picture: MatLike, region: Tuple[int, int, int, int]  # (x, y, w, h)
) -> MatLike:
    x: int = region[0]
    y: int = region[1]
    w: int = region[2]
    h: int = region[3]
    cv2.rectangle(picture, (x, y), (x + w, y + h), (67, 67, 67), 1)
    return picture


# Crop the face from the picture
def __crop_face(picture: MatLike, region: Dict[str, int]) -> MatLike:
    x: int = region["x"]
    y: int = region["y"]
    w: int = region["w"]
    h: int = region["h"]
    return picture[y : y + h, x : x + w]


# Get the matches from the dataset
# which are the closest to the detected face
def __get_face_matches(
    face: numpy.ndarray, db_path: str, model_name: str, distance_metric: str
) -> List[pandas.DataFrame]:

    matching_results = DeepFace.find(
        img_path=face,
        db_path=db_path,
        model_name=model_name,
        detector="donotdetect",  # Skip detection, we already have the face
        distance_metric=distance_metric,
    )
    return matching_results


# Process the matches and stick the matching face
# (if any)
def __process_matches(
    picture: MatLike,
    facial_area: Dict[str, Any],
    matching_results: List[pandas.DataFrame],
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
        matching_faces = DeepFace.detect_faces(
            img_path=matching_identity,
            detector=detector_backend,
            align=False,
        )
        if len(matching_faces) == 0:
            return False
    except ValueError:
        return False

    x: int = facial_area["x"]
    y: int = facial_area["y"]
    w: int = facial_area["w"]
    h: int = facial_area["h"]

    matching_face = matching_faces[0]["face"]
    matching_face = cv2.resize(matching_face, (int(h / 2.5), int(w / 2.5)))
    matching_face *= 255
    matching_face = matching_face[:, :, ::-1]

    # Stick the matching face to the original picture
    # in the lower-right corner of the boxed face
    picture[
        y + h - matching_face.shape[0] : y + h, x + w - matching_face.shape[1] : x + w
    ] = matching_face

    return True
