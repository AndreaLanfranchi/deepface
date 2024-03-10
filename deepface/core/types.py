from typing import Optional
from dataclasses import dataclass
from ctypes import (
    c_uint32 as uint32_t,
)

import numpy

from cv2.typing import MatLike

@dataclass
class RangeInt:
    """
    Represents a range of integers as [start, end]
    """
    start: uint32_t = 0
    end: uint32_t = 0

    def __init__(self, start: uint32_t, end: uint32_t):
        self.start = max(start, 0)
        self.end = max(end, 0)
        self.end = max(self.end, self.start)

@dataclass
class Point:
    x: uint32_t = 0
    y: uint32_t = 0

@dataclass
class BoxDimensions:
    width: uint32_t = 0
    height: uint32_t = 0

class InPictureFace:
    """
    This class is used to represent the result of a face detection.
    It contains the detected facial area and the confidence of the detection.
    """

    def __init__(
        self,
        detector: str,
        source: MatLike,
        y: RangeInt,
        x: RangeInt,
        left_eye: Optional[Point] = None,
        right_eye: Optional[Point] = None,
        confidence: Optional[float] = None,
    ):
        self._detector = detector
        self._source = source
        self._y = y
        self._x = x
        self._left_eye = left_eye
        self._right_eye = right_eye
        self._confidence = confidence

    @property
    def height(self) -> int:
        return self._y.end - self._y.start

    @property
    def width(self) -> int:
        return self._x.end - self._x.start

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def empty(self) -> bool:
        return self.area == 0

    @property
    def face(self) -> numpy.ndarray:
        return self._source[self._y.start : self._y.end, self._x.start : self._x.end]

    @property
    def confidence(self) -> Optional[float]:
        return self._confidence
