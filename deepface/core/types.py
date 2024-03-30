from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy
import cv2

from deepface.core.colors import *

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
        object.__setattr__(self, "end", max(self.end, self.start))

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

    Attributes:

        confidence (float): The confidence score of the detection.
        bounding_box (BoundingBox): The bounding box of the detected face.
        key_points (Optional[Dict[str, Point]]): The key points of the detected face.

            The key points are a dictionary of string keys and `Point` values.
            The keys are the names of the key points and the values are the coordinates
            of the key points relative to the whole processed image.

            The key points are optionally detected hence may be None.
            Keys are:
            "lec" (left eye center),
            "rec" (right eye center),
            "nt" (nose tip),
            "mlc" (mouth left corner),
            "mrc" (mouth right corner).
            "mc" (mouth center).

            Note: not all detectors provide all key points.
            Left and right are considered from the perspective of the person in the image.
    """

    confidence: float = field(default=0.0)
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    key_points: Optional[Dict[str, Optional[Point]]] = field(default=None)

    def __post_init__(self):
        assert isinstance(self.confidence, float)
        assert isinstance(self.bounding_box, BoundingBox)
        object.__setattr__(self, "confidence", max(self.confidence, 0.0))
        if self.bounding_box.area == 0:
            raise ValueError("Bounding box must be non-empty")

        if self.key_points is not None:
            if not isinstance(self.key_points, dict):
                raise TypeError("Key points must be a dictionary")

            # Only allow the specified keys
            allowed_keys = ["lec", "rec", "nt", "mlc", "mrc", "mc"]
            object.__setattr__(
                self,
                "key_points",
                {
                    key: value
                    for key, value in self.key_points.items()
                    if key in allowed_keys
                },
            )

            for key, value in self.key_points.items():
                if not isinstance(key, str):
                    raise TypeError("Keypoint Key must be a string")
                if value is not None and not isinstance(value, Point):
                    raise TypeError("Keypoint Value must be an Optional[Point] object")

                # Ensure that the key points are within the bounding box
                if value is not None and value not in self.bounding_box:
                    raise ValueError(f"Key point \"{key}\"={value} must be inside the bounding box {self.bounding_box}")

            # Ensure that the left and right eyes are different
            # and eventually swap them
            le = self.key_points.get("lec", None)
            re = self.key_points.get("rec", None)
            if le is not None and re is not None:
                if le == re:
                    raise ValueError("Left and right eyes must be different points")
                # Swap the left and right eyes if the left eye is to the right of the right eye
                if le.x < re.x:
                    self.key_points["lec"] = re
                    self.key_points["rec"] = le

            # Ensure that the left and right mouth are different
            # and eventually swap them
            lm = self.key_points.get("mlc", None)
            rm = self.key_points.get("mrc", None)
            if lm is not None and rm is not None:
                if lm == rm:
                    raise ValueError("Left and right mouth must be different points")
                # Swap the left and right mouth if the left mouth is to the right of the right mouth
                if lm.x < rm.x:
                    self.key_points["mlc"] = rm
                    self.key_points["mrc"] = lm

    @property
    def width(self) -> int:
        return self.bounding_box.width

    @property
    def height(self) -> int:
        return self.bounding_box.height

    def plot(
        self,
        img: numpy.ndarray,
        copy: bool = False,
        color: Tuple[int, int, int] = KBGR_COLOR_BOUNDING_BOX,
        thickness: int = 2,
        key_points: bool = False,
    ) -> numpy.ndarray:
        """
        Draw the detected face boundaries and landmarks on the image.

        Args:
            img (numpy.ndarray): The image to draw on.
            copy (bool): Whether to return the drawings on a copy of the image (default: False)
            color (Tuple[int, int, int]): BGR color code for the drawings (default: KCOLOR_BGR_CYAN)
            thickness (int): Thickness of the bounding box (default: 2)
            eyes (bool): Whether to draw eye landmarks (default: False)

        Returns:
            numpy.ndarray: The image with the detected faces plotted.

        Raises:
            TypeError: If the image is not a valid numpy array.
            ValueError: If the image is empty or the bounding box is empty.
            OverflowError: If the bounding box is out of bounds of the image.
        """

        # TODO: Maybe introduce an option for rounded corners ?
        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array for an RGB image")

        img_h, img_w = img.shape[:2]
        if img_h == 0 or img_w == 0:
            raise ValueError("Image must be non-empty")

        if self.bounding_box.empty:
            raise ValueError("Bounding box must be non-empty")

        if (
            self.bounding_box.top_right.x > img_w
            or self.bounding_box.bottom_right.y > img_h
        ):
            raise OverflowError("Bounding box is out of bounds of the image")

        if copy:
            img = img.copy()

        img = cv2.rectangle(
            img,
            (self.bounding_box.top_left.x, self.bounding_box.top_left.y),
            (self.bounding_box.bottom_right.x, self.bounding_box.bottom_right.y),
            color,
            thickness,
        )

        if key_points and self.key_points is not None and len(self.key_points) > 0:
            left_eye = self.key_points.get("lec", None)
            right_eye = self.key_points.get("rec", None)
            nose = self.key_points.get("nt", None)
            left_mouth = self.key_points.get("mlc", None)
            right_mouth = self.key_points.get("mrc", None)
            center_mouth = self.key_points.get("mc", None)
            if left_eye is not None:
                img = cv2.circle(img, left_eye.tolist(), 3, KBGR_COLOR_LEFT_EYE, thickness)
            if right_eye is not None:
                img = cv2.circle(img, right_eye.tolist(), 3, KBGR_COLOR_RIGHT_EYE, thickness)
            if left_eye is not None and right_eye is not None:
                img = cv2.line(img, left_eye.tolist(), right_eye.tolist(), color, thickness)
            if nose is not None:
                img = cv2.circle(img, nose.tolist(), 3, KBGR_COLOR_NOSE, thickness)
            if left_mouth is not None:
                img = cv2.circle(img, left_mouth.tolist(), 3, KBGR_COLOR_LEFT_MOUTH, thickness)
            if right_mouth is not None:
                img = cv2.circle(img, right_mouth.tolist(), 3, KBGR_COLOR_RIGHT_MOUTH, thickness)
            if center_mouth is not None:
                img = cv2.circle(img, center_mouth.tolist(), 3, KBGR_COLOR_CENTER_MOUTH, thickness)
            if left_mouth is not None and right_mouth is not None:
                img = cv2.line(img, left_mouth.tolist(), right_mouth.tolist(), color, thickness)

        return img

    def crop(self, img: numpy.ndarray) -> numpy.ndarray:
        """
        Returns the cropping of the detected face from the image

        Args:
            img (numpy.ndarray): The image to crop from.

        Returns:
            numpy.ndarray: The detected face as a new NumPy array.

        Raises:
            TypeError: If the image is not a valid numpy array.
            ValueError: If the image is empty or the bounding box is empty.
            OverflowError: If the bounding box is out of bounds of the image.
        """

        if not isinstance(img, numpy.ndarray) or len(img.shape) != 3:
            raise TypeError("Image must be a valid numpy array")

        img_h, img_w = img.shape[:2]
        if img_h == 0 or img_w == 0:
            raise ValueError("Image must be non-empty")
        if self.bounding_box.empty:
            raise ValueError("Bounding box must be non-empty")
        if self.bounding_box.top_right.x > img_w or self.bounding_box.bottom_right.y > img_h:
            raise OverflowError("Bounding box is out of bounds of the image")
        return img[
            self.bounding_box.top_left.y : self.bounding_box.bottom_right.y,
            self.bounding_box.top_left.x : self.bounding_box.bottom_right.x,
        ]
