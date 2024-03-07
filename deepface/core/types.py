
class RangeInt:
    """
    This class is used to represent a range of integers as [start, end]
    """

    def __init__(self, start: int, end: int):
        self.start = max(start, 0)
        self.end = max(end, 0)
        self.end = max(self.end, self.start)

    def __str__(self):
        return f"RangeInt(start={self.start}, end={self.end})"

    def __repr__(self):
        return self.__str__()

class Point:
    """
    This class is used to represent a point in a 2D space
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __repr__(self):
        return self.__str__()
