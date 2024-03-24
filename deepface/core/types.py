from typing import Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy
import cv2


@dataclass(frozen=True)
class RangeInt:
    """
    Represents a range of integers as [start, end]
    """

    start: int = field(default=int(0))
    end: int = field(default=int(0))

    def __post_init__(self):
        assert isinstance(self.start, int)
        assert isinstance(self.end, int)
        object.__setattr__(self, "start", max(self.start, 0))
        object.__setattr__(self, "end", max(self.end, 0))
        assert self.start <= self.end

    @property
    def span(self) -> int:
        return self.end - self.start

    def contains(self, item: int) -> bool:
        return self.start <= item <= self.end


@dataclass(frozen=True)
class Point:
    """
    Represents a point in a 2D plane.
    Negative values are normalized to 0.
    """

    x: int = field(default=0)
    y: int = field(default=0)

    def __post_init__(self):
        assert isinstance(self.x, int)
        assert isinstance(self.y, int)
        object.__setattr__(self, "x", max(self.x, 0))
        object.__setattr__(self, "y", max(self.y, 0))

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

    def __eq__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __ne__(self, other: "Point") -> bool:
        return not self.__eq__(other)

    def __gt__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self.x > other.x and self.y > other.y
        return False

    def __ge__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self.x >= other.x and self.y >= other.y
        return False

    def __lt__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self.x < other.x and self.y < other.y
        return False

    def __le__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self.x <= other.x and self.y <= other.y
        return False

    def tolist(self) -> list:
        return [self.x, self.y]


@dataclass(frozen=True)
class BoundingBox:
    """
    Represents a box in a 2D plane enclosed by two points which
    are the top-left and bottom-right corners of the box.
    The top-right and bottom-left corners are also provided
    as derived properties for convenience.
    Is used to represent the position of a face in an image.
    """

    top_left: Point = field(default_factory=Point)
    bottom_right: Point = field(default_factory=Point)

    def __post_init__(self):
        assert isinstance(self.top_left, Point)
        assert isinstance(self.bottom_right, Point)
        if (self.top_left <= self.bottom_right) == False:
            raise ValueError("Top-left must be less than or equal to bottom-right")

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y

    @property
    def top_right(self) -> Point:
        return Point(x=self.bottom_right.x, y=self.top_left.y)

    @property
    def bottom_left(self) -> Point:
        return Point(x=self.top_left.x, y=self.bottom_right.y)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def empty(self) -> bool:
        return self.area == 0

    @property
    def center(self) -> Point:
        return Point(
            x=self.top_left.x + self.width // 2,
            y=self.top_left.y + self.height // 2,
        )

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        return (self.top_left.x, self.top_left.y, self.width, self.height)

    def __contains__(self, point: Point) -> bool:
        return (
            self.top_left.x <= point.x <= self.bottom_right.x
            and self.top_left.y <= point.y <= self.bottom_right.y
        )


@dataclass(frozen=True)
class BoxDimensions:

    width: int = field(default=0)
    height: int = field(default=0)

    def __post_init__(self):
        assert isinstance(self.width, int)
        assert isinstance(self.height, int)
        object.__setattr__(self, "width", max(self.width, 0))
        object.__setattr__(self, "height", max(self.height, 0))

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

    def __lt__(self, other: Union[BoundingBox, "BoxDimensions"]) -> bool:
        if isinstance(other, BoundingBox):
            return self.width < other.width or self.height < other.height
        if isinstance(other, BoxDimensions):
            return self.width < other.width or self.height < other.height
        return False


@dataclass(frozen=True)
class DetectedFace:
    """
    Represents the detection of a single face in an image.
    Note: an image may contain multiple faces.
    """

    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    left_eye: Optional[Point] = field(default=None)
    right_eye: Optional[Point] = field(default=None)
    confidence: Optional[float] = field(default=None)

    def __post_init__(self):
        assert isinstance(self.bounding_box, BoundingBox)
        assert self.left_eye is None or isinstance(self.left_eye, Point)
        assert self.right_eye is None or isinstance(self.right_eye, Point)
        if isinstance(self.confidence, float):
            object.__setattr__(self, "confidence", max(self.confidence, 0.0))
        else:
            object.__setattr__(self, "confidence", float(0.0))

        if self.bounding_box.area == 0:
            raise ValueError("Bounding box must be non-empty")

        if bool(self.left_eye is None) != bool(self.right_eye is None):
            raise ValueError("Both eyes must be provided or both must be None")

        if self.left_eye is not None and self.right_eye is not None:
            if self.left_eye == self.right_eye:
                raise ValueError("Left and right eyes must be different")
            if self.left_eye > self.right_eye:
                raise ValueError("Left eye must be to the left of the right eye")
            if self.left_eye not in self.bounding_box:
                raise ValueError("Left eye must be inside the bounding box")
            if self.right_eye not in self.bounding_box:
                raise ValueError("Right eye must be inside the bounding box")

    @property
    def width(self) -> int:
        return self.bounding_box.width

    @property
    def height(self) -> int:
        return self.bounding_box.height

    def plot(
        self,
        img: numpy.ndarray,
        color: Tuple[int, int, int] = (255, 255, 224),  # BGR light cyan
        thickness: int = 2,
        eyes: bool = False,
    ) -> numpy.ndarray:
        """
        Draw the bounding box and eyes on the image.
        """
        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array")
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Image must be non-empty")
        if self.bounding_box.empty:
            raise ValueError("Bounding box must be non-empty")
        if eyes == True and (self.left_eye is not None and self.right_eye is not None):
            img = cv2.circle(img, self.left_eye.tolist(), 5, color, thickness)
            img = cv2.circle(img, self.right_eye.tolist(), 5, color, thickness)
            img = cv2.line(
                img, self.left_eye.tolist(), self.right_eye.tolist(), color, thickness
            )
        img = cv2.rectangle(
            img,
            (self.bounding_box.top_left.x, self.bounding_box.top_left.y),
            (self.bounding_box.bottom_right.x, self.bounding_box.bottom_right.y),
            color,
            thickness,
        )
        return img

    def crop(self, img: numpy.ndarray) -> numpy.ndarray:
        """
        Crop the face from the image.
        """
        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array")
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Image must be non-empty")
        if self.bounding_box.empty:
            raise ValueError("Bounding box must be non-empty")
        return img[
            self.bounding_box.top_left.y : self.bounding_box.bottom_right.y,
            self.bounding_box.top_left.x : self.bounding_box.bottom_right.x,
        ]
