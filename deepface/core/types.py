from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy
import cv2

from deepface.core.colors import (
    KBGR_COLOR_BOUNDING_BOX,
    KBGR_COLOR_CENTER_MOUTH,
    KBGR_COLOR_LEFT_EYE,
    KBGR_COLOR_LEFT_MOUTH,
    KBGR_COLOR_NOSE,
    KBGR_COLOR_RIGHT_EYE,
    KBGR_COLOR_RIGHT_MOUTH,
)

from deepface.core.imgutils import is_valid_image


@dataclass(frozen=True)
class RangeInt:
    """
    Represents a range of integers as [start, end]\n
    Negative values are normalized to 0.\n
    The start value is always less than or equal to the end value.\n
    Used to represent the range of a sequence of coordinates on a 2D plane.

    Attributes:
    -----------

        start (int): The start of the range.
        end (int): The end of the range.

    Properties:
    -----------

        span (int): The span of the range.
        empty (bool): Whether the range is empty.
        contains(item: int) -> bool: Whether the range contains the item.
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

    def contains(self, value: int) -> bool:
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        return self.start <= value <= self.end


@dataclass(frozen=True)
class Point:
    """
    Represents a point in a 2D plane.\n
    Negative values are normalized to 0.

    Attributes:
    -----------

        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.

    """

    x: int = field(default=int(0))
    y: int = field(default=int(0))

    def __post_init__(self):
        assert isinstance(self.x, int)
        assert isinstance(self.y, int)
        object.__setattr__(self, "x", max(self.x, int(0)))
        object.__setattr__(self, "y", max(self.y, int(0)))

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __getitem__(self, index: int) -> int:
        match index:
            case 0:
                return self.x
            case 1:
                return self.y
            case _:
                raise IndexError("Index out of range")

    def __eq__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __ne__(self, other: "Point") -> bool:
        return not self.__eq__(other)

    def __gt__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            if self.x == other.x:
                return self.y > other.y
            return self.x > other.x
        return False

    def __ge__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self > other or self == other
        return False

    def __lt__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            if self.x == other.x:
                return self.y < other.y
            return self.x < other.x
        return False

    def __le__(self, other: "Point") -> bool:
        if isinstance(other, Point):
            return self < other or self == other
        return False

    def tolist(self) -> list:
        return [self.x, self.y]


@dataclass(frozen=True)
class BoundingBox:
    """
    Represents a box in a 2D plane enclosed by two points which
    are the top-left and bottom-right corners of the box.\n
    The top-right and bottom-left corners are also provided
    as derived properties for convenience.\n
    Is used to represent the position of a face in an image.

    Attributes:
    -----------

        top_left (Point): The top-left corner of the box.
        bottom_right (Point): The bottom-right corner of the box.

    Properties:
    -----------

        width (int): The width of the box.
        height (int): The height of the box.
        top_right (Point): The top-right corner of the box.
        bottom_left (Point): The bottom-left corner of the box.
        area (int): The area of the box.
        empty (bool): Whether the box is empty.
        center (Point): The center of the box.
        xywh (Tuple[int, int, int, int]): The x, y, width, height of the box.

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
        return (
            self.top_left.x,
            self.top_left.y,
            self.width,
            self.height,
        )

    @property
    def x1y1x2y2(self) -> Tuple[int, int, int, int]:
        return (
            self.top_left.x,
            self.top_left.y,
            self.bottom_right.x,
            self.bottom_right.y,
        )

    def __contains__(self, point: Point) -> bool:
        return (
            self.top_left.x <= point.x <= self.bottom_right.x
            and self.top_left.y <= point.y <= self.bottom_right.y
        )


@dataclass(frozen=True)
class BoxDimensions:
    """
    Represents the dimensions of a box in a 2D plane.\n
    Used to represent the width and height of a face's bounding box.

    Attributes:
    -----------

        width (int): The width of the box.
        height (int): The height of the box.
    """

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
        match index:
            case 0:
                return self.width
            case 1:
                return self.height
            case _:
                raise IndexError("Index out of range")

    @property
    def area(self) -> int:
        return self.width * self.height

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
    -----------

        confidence (float): The confidence score of the detection.
        bounding_box (BoundingBox): The bounding box of the detected face.
        key_points (Optional[Dict[str, Point]]): The key points of the detected face.
        embeddings (Optional[List[float]]): The embeddings of the detected face.
            Embeddings are optionally provided by the extractor.

    Notes:
    ------

        The key points are a dictionary of string keys and class `Point` values.
        The keys are the names of the key points and the values are the coordinates
        of the key points relative to the whole processed image.
        The key points are optionally detected hence may be None.\n
        Keys are:
        * "lec" (left eye center),
        * "rec" (right eye center),
        * "nt" (nose tip),
        * "mlc" (mouth left corner),
        * "mrc" (mouth right corner).
        * "mc" (mouth center).

        Not all detectors provide all key points.\n
        Left and right are considered from the perspective of the person in the image.
    """

    confidence: float = field(default=0.0)
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    key_points: Optional[Dict[str, Point]] = field(default=None)
    embeddings: Optional[List[float]] = field(default=None)
    attributes: Optional[Dict[str, str]] = field(default=None)

    def __post_init__(self):
        assert isinstance(self.confidence, float)
        assert isinstance(self.bounding_box, BoundingBox)
        object.__setattr__(self, "confidence", max(self.confidence, 0.0))
        if self.bounding_box.area == 0:
            raise ValueError("Bounding box must be non-empty")

        if self.key_points is not None:
            if not isinstance(self.key_points, dict):
                raise TypeError("Key points must be a dictionary")

            if len(self.key_points) == 0:
                raise ValueError("Key points must be non-empty or None")

            # Only allow the specified keys
            allowed_keys: List[str] = ["lec", "rec", "nt", "mlc", "mrc", "mc"]
            for key, value in self.key_points.items():
                if not isinstance(key, str):
                    what: str = "Keypoint Key must be a string"
                    what += f" : got {type(key)}=[{key}]"
                    raise TypeError(what)
                if key not in allowed_keys:
                    what: str = f'Keypoint Key "{key}" is not one of'
                    what += f" the allowed keys {allowed_keys}"
                    raise ValueError(what)
                if not isinstance(value, Point):
                    raise TypeError(
                        "Keypoint Value must be a Point object : got {type(value)}"
                    )
                if value not in self.bounding_box:
                    what: str = f'Key point "{key}"={value} is not inside'
                    what += f" the bounding box xywh [{self.bounding_box.xywh}]"
                    raise ValueError(what)

            le = self.key_points.get("lec", None)
            re = self.key_points.get("rec", None)
            if le is not None and re is not None:
                if le == re:
                    what: str = "Left and right eyes must be different points."
                    what += f" Got left eye={le} and right eye={re}"
                    raise ValueError(what)
                # Swap the left and right eyes if the left eye is to the right of the right eye
                if le.x < re.x:
                    self.key_points["lec"] = re
                    self.key_points["rec"] = le

            lm = self.key_points.get("mlc", None)
            rm = self.key_points.get("mrc", None)
            if lm is not None and rm is not None:
                if lm == rm:
                    what: str = "Left and right mouth corners must be different points."
                    what += f" Got left mouth={lm} and right mouth={rm}"
                    raise ValueError(what)
                # Swap the left and right mouth if the left mouth is to the right of the right mouth
                if lm.x < rm.x:
                    self.key_points["mlc"] = rm
                    self.key_points["mrc"] = lm

        if self.embeddings is not None:
            if not isinstance(self.embeddings, list):
                raise TypeError("Embeddings must be a list")
            if len(self.embeddings) == 0:
                raise ValueError("Embeddings must be non-empty list")
            for value in self.embeddings:
                if not isinstance(value, float):
                    raise TypeError("Embedding value must be a float")

        if self.attributes is not None:
            if not isinstance(self.attributes, dict):
                raise TypeError("Attributes must be a dictionary")
            if len(self.attributes) == 0:
                raise ValueError("Attributes must be non-empty dictionary")
            for key, value in self.attributes.items():
                if not isinstance(key, str):
                    raise TypeError("Attribute Key must be a string")
                if not isinstance(value, str):
                    raise TypeError("Attribute Value must be a string")

    def set_key_points(self, key_points: Dict[str, Point]):
        """
        Set the key points of the detected face.

        Args:
        ----
            key_points (Dict[str, Point]): The key points of the detected face.

        Raises:
        -------
            TypeError: If the key points are not a list of Point objects.
            ValueError: If the key points are empty or not the correct length.
        """

        if not isinstance(key_points, dict):
            raise TypeError("Key points must be a dictionary")
        if len(key_points) == 0:
            return

        allowed_keys: List[str] = ["lec", "rec", "nt", "mlc", "mrc", "mc"]

        for key, value in key_points.items():
            if not isinstance(key, str):
                raise TypeError("Key point Key must be a string")
            if key not in allowed_keys:
                what: str = f'Key point Key "{key}" is not one of the allowed keys'
                what += f" {allowed_keys}"
                raise ValueError(what)
            if not isinstance(value, Point):
                raise TypeError("Key point Value must be a Point object")
            if value not in self.bounding_box:
                what: str = f'Key point "{key}"={value} is not inside'
                what += f" the bounding box xywh [{self.bounding_box.xywh}]"
                raise ValueError(what)

        le = key_points.get("lec", None)
        re = key_points.get("rec", None)
        if le is not None and re is not None:
            if le == re:
                what: str = "Left and right eyes must be different points."
                what += f" Got left eye={le} and right eye={re}"
                raise ValueError(what)
            # Swap the left and right eyes if the left eye is to the right of the right eye
            if le.x < re.x:
                key_points["lec"] = re
                key_points["rec"] = le

        lm = key_points.get("mlc", None)
        rm = key_points.get("mrc", None)
        if lm is not None and rm is not None:
            if lm == rm:
                what: str = "Left and right mouth corners must be different points."
                what += f" Got left mouth={lm} and right mouth={rm}"
                raise ValueError(what)
            # Swap the left and right mouth if the left mouth is to the right of the right mouth
            if lm.x < rm.x:
                key_points["mlc"] = rm
                key_points["mrc"] = lm

        object.__setattr__(self, "key_points", key_points)

    def set_embeddings(self, embeddings: List[float]):
        """
        Set the embeddings of the detected face.

        Args:
        ----
            embeddings (List[float]): The embeddings of the detected face.

        Raises:
        -------
            TypeError: If the embeddings are not a list of floats.
            ValueError: If the embeddings are empty.
        """

        if not isinstance(embeddings, list):
            raise TypeError("Embeddings must be a list")
        if len(embeddings) == 0:
            return
        for value in embeddings:
            if not isinstance(value, float):
                raise TypeError("Embedding value must be a float")
        object.__setattr__(self, "embeddings", embeddings)

    def set_attributes(self, attributes: Dict[str, str]):
        """
        Set the attributes of the detected face.

        Args:
        ----
            attributes (Dict[str, str]): The attributes of the detected face.

        Raises:
        -------
            TypeError: If the attributes are not a dictionary.
            ValueError: If the attributes are empty.
        """

        if not isinstance(attributes, dict):
            raise TypeError("Attributes must be a dictionary")
        if len(attributes) == 0:
            return
        for key, value in attributes.items():
            if not isinstance(key, str):
                raise TypeError("Attribute Key must be a string")
            if not isinstance(value, str):
                raise TypeError("Attribute Value must be a string")
        
        if self.attributes is not None:
            self.attributes.update(attributes)
        else:
            object.__setattr__(self, "attributes", attributes)

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

        Params:
        -------
            img (numpy.ndarray): The image to draw on.
            copy (bool): Whether to return the drawings on a copy of the image (default: False)
            color (Tuple[int, int, int]): BGR color code for the drawings (default: KCOLOR_BGR_CYAN)
            thickness (int): Thickness of the bounding box (default: 2)
            key_points (bool): Whether to draw eye landmarks (default: False)

        Returns:
        --------
            numpy.ndarray: The (copy of) image with the detected face plotted on it.

        Raises:
        -------
            TypeError: If the image is not a valid numpy array.
            ValueError: If the image is empty or the bounding box is empty.
            OverflowError: If the bounding box is out of bounds of the image.
        """

        if not is_valid_image(img):
            raise TypeError("Image must be a valid numpy array for a non empty image")

        if self.bounding_box.empty:
            raise ValueError("Bounding box must be non-empty")

        img_h, img_w = img.shape[:2]
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
                img = cv2.circle(
                    img, left_eye.tolist(), 3, KBGR_COLOR_LEFT_EYE, thickness
                )
            if right_eye is not None:
                img = cv2.circle(
                    img, right_eye.tolist(), 3, KBGR_COLOR_RIGHT_EYE, thickness
                )
            if left_eye is not None and right_eye is not None:
                img = cv2.line(
                    img, left_eye.tolist(), right_eye.tolist(), color, thickness
                )
            if nose is not None:
                img = cv2.circle(img, nose.tolist(), 3, KBGR_COLOR_NOSE, thickness)
            if left_mouth is not None:
                img = cv2.circle(
                    img, left_mouth.tolist(), 3, KBGR_COLOR_LEFT_MOUTH, thickness
                )
            if right_mouth is not None:
                img = cv2.circle(
                    img, right_mouth.tolist(), 3, KBGR_COLOR_RIGHT_MOUTH, thickness
                )
            if center_mouth is not None:
                img = cv2.circle(
                    img, center_mouth.tolist(), 3, KBGR_COLOR_CENTER_MOUTH, thickness
                )
            if left_mouth is not None and right_mouth is not None:
                img = cv2.line(
                    img, left_mouth.tolist(), right_mouth.tolist(), color, thickness
                )

        return img

    def crop(self, img: numpy.ndarray) -> numpy.ndarray:
        """
        Returns the cropping of the detected face from the image

        Args:
        -----
            img (numpy.ndarray): The image to crop from.

        Returns:
        --------
            numpy.ndarray: The cropped image from the bounding box.

        Raises:
        -------
            TypeError: If the image is not a valid numpy array.
            ValueError: If the image is empty or the bounding box is empty.
            OverflowError: If the bounding box is out of bounds of the image.
        """

        if not is_valid_image(img):
            raise TypeError("Image must be a valid numpy array for a non empty image")

        img_h, img_w, *_ = img.shape
        if (
            self.bounding_box.top_right.x > img_w
            or self.bounding_box.bottom_right.y > img_h
        ):
            what: str = "Bounding box exceeds image dimensions: \n"
            what += f" x [{self.bounding_box.top_right.x}] : y [{self.bounding_box.bottom_right.y}] \n"
            what += f" max x [{img_w}] : max y [{img_h}]"
            raise OverflowError(what)

        cropped = img[
            self.bounding_box.top_left.y : self.bounding_box.bottom_right.y,
            self.bounding_box.top_left.x : self.bounding_box.bottom_right.x,
        ]
        return cropped
