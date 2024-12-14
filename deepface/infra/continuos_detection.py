import os
import hashlib
import pickle
import threading
from typing import Dict, List, Optional, Union

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from deepface.core import imgutils
from deepface.core.analyzer import Analyzer
from deepface.core.detector import Detector
from deepface.core.extractor import Extractor
from deepface.core.types import BoxDimensions, DetectedFace
from deepface.infra import detection
from deepface.commons.logger import Logger

logger = Logger.get_instance()


class ContinuosFaceDetection:

    _data_root: str
    _pickle_file: str
    _detector_instance: Detector
    _min_confidence: Optional[float]
    _min_dims: Optional[BoxDimensions]
    _key_points: bool
    _extractor_instance: Optional[Extractor] = None
    _attributes: Optional[List[str]] = None

    _evt_handler = FileSystemEventHandler()
    _observer = Observer()

    _data: Dict[str, Optional[List[DetectedFace]]] = {}

    _lock = threading.Lock()

    def __init__(
        self,
        data_root: str,
        detector: Union[str, Detector] = "default",
        min_confidence: Optional[float] = None,
        min_dims: Optional[BoxDimensions] = None,
        key_points: bool = True,
        extractor: Optional[Union[str, Extractor]] = None,
        attributes: Optional[List[str]] = None,
    ):

        # Validate data_root
        if not isinstance(data_root, str):
            raise TypeError("Argument data_root must be a valid directory path")
        if not os.path.isdir(data_root):
            raise ValueError(f"Directory {data_root} does not exist")

        self._data_root = os.path.abspath(data_root)

        self._detector_instance = Detector.instance(detector)
        self._extractor_instance = Extractor.instance(extractor) if extractor else None
        self._min_confidence = min_confidence
        self._min_dims = min_dims
        self._key_points = key_points
        self._attributes = attributes

        if attributes is not None:
            if isinstance(attributes, str):
                attributes = [
                    attributes,
                ]
            if not isinstance(attributes, list):
                what: str = (
                    "Invalid attributes argument type. Expected [List[str] | None]"
                )
                what += f" but got {type(attributes).__name__}"
                raise TypeError(what)

            for attribute in attributes:
                _ = Analyzer.instance(attribute)

        self._init_piclke_file()
        self._load_pickle()

    def _init_piclke_file(self):

        input_values: str = self._detector_instance.name
        if self._min_confidence:
            input_values += str(self._min_confidence)
        if self._min_dims:
            input_values += str(self._min_dims)
        if self._key_points:
            input_values += "keypoints"
        if self._extractor_instance:
            input_values += self._extractor_instance.name
        if self._attributes:
            input_values += str(self._attributes)

        hashed_values = hashlib.md5(input_values.encode()).hexdigest()
        self._pickle_file = os.path.join(self._data_root, f"_dfstore_{hashed_values}_.pkl")
        logger.info(f"Face detection store file: {self._pickle_file}")

    def _load_pickle(self):

        if not os.path.exists(self._pickle_file):
            with open(self._pickle_file, "wb") as f:
                pickle.dump(self._data, f)

        try:
            with open(self._pickle_file, "rb") as f:
                self._data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading face detection store: {e}")
            os.remove(self._pickle_file)
            self._data = {}

        # Get all images files in the directory and subdirs
        image_files = imgutils.get_all_image_files(self._data_root, recurse=True, check_ext=True)
        new_files = set(image_files) - set(self._data.keys())
        del_files = set(self._data.keys()) - set(image_files)
        logger.info(f"New image files: {len(new_files)} - Removed image files: {len(del_files)}")
        must_update: bool = False

        if len(del_files) > 0:
            must_update = True
            for old_file in del_files:
                del self._data[old_file]

        if len(new_files) > 0:
            must_update = True
            self._data.update(
                detection.batch_detect_faces(
                    inputs=list(new_files),
                    detector=self._detector_instance,
                    min_confidence=self._min_confidence,
                    min_dims=self._min_dims,
                    key_points=self._key_points,
                    raise_notfound=False,
                    extractor=self._extractor_instance,
                    attributes=self._attributes,
                )
            )

        if must_update:
            with open(self._pickle_file, "wb") as f:
                pickle.dump(self._data, f)

        # Count all faces detected
        total_faces = 0
        for _, detections in self._data.items():
            if detections is not None:
                total_faces += len(detections)

        logger.info(f"Face detection store loaded with {len(self._data)} images and {total_faces} faces detected")

    def _on_entry_created(self, event):
        what: str = "Directory" if event.is_directory else "File"
        logger.debug(f"{what} created : {event.src_path}")

        # The creation of files is immediately detected by watchdog
        # on their creation, but the file may not be ready to be read
        # by the detector. We need to wait until the file is closed
        # before we can read it. Therefore, we need to wait until the
        # _on_entry_modified event is triggered
        # Keeping this for reference

    def _on_entry_deleted(self, event):

        what: str = "Directory" if event.is_directory else "File"
        logger.debug(f"{what} deleted : {event.src_path}")

        if event.is_directory:
            # Remove all entries from the database
            # where the file path starts with this directory
            # path

            directory_path: str = str(event.src_path)
            if not directory_path.endswith(str(os.pathsep)):
                directory_path += str(os.pathsep)

            with self._lock:
                keys_to_delete = [
                    key for key in self._data.keys() if key.startswith(directory_path)
                ]
                for key in keys_to_delete:
                    del self._data[key]
                logger.debug(f"{len(keys_to_delete)} entries removed")

        else:
            with self._lock:
                if event.src_path in self._data:
                    del self._data[event.src_path]

    def _on_entry_modified(self, event):
        what: str = "Directory" if event.is_directory else "File"
        logger.debug(f"{what} modified : {event.src_path}")

    def _on_entry_moved(self, event):
        what: str = "Directory" if event.is_directory else "File"
        logger.debug(f"{what} moved : {event.src_path} -> {event.dest_path}")

    def start(self):
        self._start_monitoring()

    def stop(self):
        self._observer.stop()
        self._observer.join()
        logger.info(f"Continuos face detection stopped on directory {self._data_root}")

    def _start_monitoring(self):

        self._evt_handler.on_created = self._on_entry_created
        self._evt_handler.on_deleted = self._on_entry_deleted
        self._evt_handler.on_modified = self._on_entry_modified
        self._evt_handler.on_moved = self._on_entry_moved

        self._observer.schedule(self._evt_handler, self._data_root, recursive=True)
        self._observer.start()
        logger.info(f"Continuos face detection started on directory {self._data_root}")
