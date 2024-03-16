from typing import Optional
from dataclasses import dataclass

import numpy

from cv2.typing import MatLike


@dataclass
class RangeInt:
    """
    Represents a range of integers as [start, end]
    """

    start: int = 0
    end: int = 0

    def __init__(self, start: int, end: int):
        self.start = max(start, 0)
        self.end = max(end, 0)
        self.end = max(self.end, self.start)

    @property
    def span(self) -> int:
        return self.end - self.start

    def contains(self, item: int) -> bool:
        return self.start <= item <= self.end


@dataclass(frozen=True)
class Point:
    x: int = 0
    y: int = 0

    def __init__(self, x: int, y: int):
        assert isinstance(x, int)
        assert isinstance(y, int)
        self.x = max(x, 0)
        self.y = max(y, 0)

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError("Index out of range")


@dataclass(frozen=True)
class BoxDimensions:
    width: int = 0
    height: int = 0

    def __init__(self, width: int, height: int):
        assert isinstance(width, int)
        assert isinstance(height, int)
        self.width = max(width, 0)
        self.height = max(height, 0)

    def __iter__(self):
        yield self.width
        yield self.height

    def __len__(self):
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.width
        if index == 1:
            return self.height
        raise IndexError("Index out of range")

@dataclass(frozen=True)
class InPictureFace:
    """
    Represents the result of a face detection.
    """

    def __init__(
        self,
        detector: str,  # The name of the detector used
        source: MatLike,  # The original image being processed
        y_range: RangeInt,  # The vertical range for the box containing the face
        x_range: RangeInt,  # The horizontal range for the box containing the face
        left_eye: Optional[Point] = None,  # The coordinates of the left eye (if any)
        right_eye: Optional[Point] = None,  # The coordinates of the right eye (if any)
        confidence: Optional[float] = None,  # The confidence of the detection (if any)
    ):
        assert isinstance(detector, str)
        assert isinstance(source, MatLike)
        self._detector = detector.strip()
        self._source = source
        self._y_range = y_range
        self._x_range = x_range
        self._left_eye = left_eye
        self._right_eye = right_eye
        self._confidence = confidence

    @property
    def detector(self) -> str:
        return self._detector

    @property
    def source(self) -> MatLike:
        return self._source

    @property
    def top_left(self) -> Point:
        return Point(self._x_range.start, self._y_range.start)
    
    @property
    def bottom_right(self) -> Point:
        return Point(self._x_range.end, self._y_range.end)

    @property
    def height(self) -> int:
        return self._y_range.end - self._y_range.start

    @property
    def width(self) -> int:
        return self._x_range.end - self._x_range.start

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def empty(self) -> bool:
        return self.area == 0

    @property
    def crop_face(self) -> numpy.ndarray:
        return self._source[
            self._y_range.start : self._y_range.end,
            self._x_range.start : self._x_range.end,
        ]

    @property
    def confidence(self) -> Optional[float]:
        return self._confidence
