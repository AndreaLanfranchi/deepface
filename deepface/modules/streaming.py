from typing import Any, Dict, List, Tuple, Union, Optional

import sys
import os
import time
import threading
import traceback

import numpy
import cv2
from cv2.typing import MatLike
import pandas

from deepface import DeepFace
from deepface.core.analyzer import Analyzer
from deepface.core.colors import KBGR_COLOR_WHITE
from deepface.core.detector import Detector
from deepface.core.extractor import Extractor
from deepface.core.exceptions import FaceNotFoundError
from deepface.core.types import BoxDimensions, DetectedFace
from deepface.commons.logger import Logger

logger = Logger.get_instance()

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class Stream(threading.Thread):
    """
    A class to read frames from a video capture source in a separate thread.

    Rationale:
    ---------
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
    detector: Union[str, Detector] = "default",
    extractor: Union[str, Extractor] = "default",
    distance_metric: str = "cosine",
    attributes: Optional[List[str]] = None,
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
    extractor = Extractor.instance(extractor)

    # Constants
    capture_window_title: str = "Capture"

    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays

    # Lazy load the attributes analyzers
    if attributes is not None:
        for i in range(len(attributes) - 1, -1, -1):
            try:
                _ = Analyzer.instance(attributes[i])
            except Exception as ex:
                logger.warn(f"Invalid attribute [{attributes[i]}] :{ex.args[0]}")
                attributes.pop(i)

    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    # _ = DeepFace.find(
    #     img=numpy.zeros((1, 1), dtype=numpy.uint8),
    #     db_path=db_path,
    #     detector=detector,
    #     extractor=extractor,
    #     distance_metric=distance_metric,
    # )
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
    # 3. The best capture eventually undergoes facial recognition
    #    which will produce the stiching of the matching face(s)
    #    to the boxed detected face(s) in the best capture.
    # 4. The best capture (altered) is then displayed for a few
    #    seconds with a countodown box till time_threshold is reached.
    #
    # The whole loop gracefully exits when the user presses 'q' or
    # the capture source is closed/failed.

    # For each good capture, store the image and the analysis result
    good_captures: List[Detector.Results] = []

    while True:
        try:
            capture_successful, captured_frame = stream.read()
            if not capture_successful:
                raise IOError("Capture from source failed")
            if captured_frame.size == 0:
                continue

            cv2.imshow(capture_window_title, captured_frame)
            _cv2_refresh()

            _process_frame(
                frame=captured_frame,
                faces_count_threshold=faces_count_threshold,
                detector=detector,
                good_captures=good_captures,
            )

            if len(good_captures) < valid_frames_count:
                continue

            best_frame, best_detection = _get_best_detection(good_captures)
            boxed_frame = best_detection.plot(
                img=best_frame,
                copy=True,
                thickness=1,
                key_points=True,
            )
            cv2.imshow(
                capture_window_title,
                boxed_frame,
            )
            _cv2_refresh()

            # TODO : Perform facial attribute analysis and face recognition

            # Display the results
            should_freeze = True
            # for item in best_faces:

            #     # Detect matches for this face
            #     matching_results = _get_face_matches(
            #         face=item["face"],
            #         db_path=db_path,
            #         model_name=decomposer,
            #         distance_metric=distance_metric,
            #     )
            #     if len(matching_results) > 0:
            #         # Applies matches to the best capture
            #         if (
            #             _process_matches(
            #                 best_capture,
            #                 item["facial_area"],
            #                 matching_results,
            #                 detector,
            #             )
            #             == True
            #         ):
            #             should_freeze = True
            #             _cv2_refresh()

            if should_freeze:
                # Count back from time_threshold to 0
                # Display the best capture for a few seconds
                # with a counting box
                for i in range(freeze_time_seconds, 0, -1):
                    cv2.rectangle(boxed_frame, (10, 10), (70, 50), (67, 67, 67), -10)
                    cv2.putText(
                        boxed_frame,
                        str(i),
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        KBGR_COLOR_WHITE,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(capture_window_title, boxed_frame)
                    _cv2_refresh(1000)

            good_captures.clear()

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


def _cv2_refresh(timeout: int = 1):
    """
    Refresh the OpenCV displayed window(s) and check for user input

    Params:
    -------
    timeout: int
        Time in milliseconds to wait for a key press

    Raises:
    -------
    KeyboardInterrupt
        If the user presses the 'q' key
    """
    timeout = max(1, timeout)
    result: int = cv2.waitKey(timeout) & 0xFF
    if result == ord("q"):
        raise KeyboardInterrupt("User requested to stop")


def _process_frame(
    frame: MatLike,
    detector: Detector,
    good_captures: List[Detector.Results],
    faces_count_threshold: int,
    min_box_dims: BoxDimensions = BoxDimensions(40, 40),
):
    """
    Process the frame to detect faces and store the good captures

    Params:
    -------
    frame: numpy.ndarray
        Image to process
    detector: Detector
        Face detector to use
    faces_count_threshold: int
        Maximum number of faces allowed in the frame
    min_box_dims: BoxDimensions
        Minimum dimensions for a face bounding box
    good_captures: List[Detector.Results]
        List of good captures where to store the results

    """
    try:
        start_time = time.time()
        detector_results: Detector.Results = detector.process(
            img=frame,
            tag="Capture",
            min_dims=min_box_dims,
            key_points=True,
            raise_notfound=True,
        )
        debug_line = f"Frame detection time: {(time.time() - start_time):.5f} seconds."
        debug_line += f" Got {len(detector_results)} face(s)"
        logger.debug(debug_line)

        # Treat too many results as error
        if len(detector_results) > faces_count_threshold:
            raise OverflowError(f"Too many faces found: {len(detector_results)}")

        # Store the good capture and its detection result
        good_captures.append(detector_results)

    # We only catch the FaceNotFoundError, ValueError and OverflowError exceptions here
    # to reset the good_captures list. Other exceptions are not caught here and will pop
    # to the caller.
    except FaceNotFoundError as e:
        logger.debug(f"Error : {e}")
        good_captures.clear()
    except ValueError as e:
        logger.debug(f"Error : {e}")
        good_captures.clear()
    except OverflowError as e:
        logger.debug(f"Error : {e}")
        good_captures.clear()


# Get the best captured face from the list of good captures
# The best capture is the one with the largest facial area
# and, in case of equality, the one with the highest
# confidence level.
def _get_best_detection(
    good_captures: List[Detector.Results],
) -> Tuple[numpy.ndarray, DetectedFace]:
    best_area: int = 0
    best_confidence: float = 0
    best_i: int = 0
    best_j: int = 0

    for i, outcome in enumerate(good_captures):
        for j, face in enumerate(outcome.detections):
            area = face.bounding_box.area
            confidence = face.confidence if face.confidence is not None else float(0)
            if area > best_area or (area == best_area and confidence > best_confidence):
                best_area = area
                best_confidence = confidence
                best_i = i
                best_j = j

    return (good_captures[best_i].img, good_captures[best_i].detections[best_j])


# Get the matches from the dataset
# which are the closest to the detected face
def _get_face_matches(
    face: numpy.ndarray, db_path: str, model_name: str, distance_metric: str
) -> List[pandas.DataFrame]:

    matching_results = DeepFace.find(
        img=face,
        db_path=db_path,
        extractor=model_name,
        detector="donotdetect",  # Skip detection, we already have the face
        distance_metric=distance_metric,
    )
    return matching_results


# Process the matches and stick the matching face
# (if any)
def _process_matches(
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
