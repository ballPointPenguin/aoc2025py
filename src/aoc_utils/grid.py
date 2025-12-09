"""Grid utilities for 2D puzzle problems."""

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point with common operations."""

    x: int
    y: int

    def __add__(self, other: Self) -> Self:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: int) -> Self:
        return Point(self.x * scalar, self.y * scalar)

    def manhattan(self, other: Self) -> int:
        """Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def neighbors4(self) -> Iterator[Self]:
        """Yield the 4 cardinal neighbors."""
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            yield Point(self.x + dx, self.y + dy)

    def neighbors8(self) -> Iterator[Self]:
        """Yield all 8 neighbors (including diagonals)."""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    yield Point(self.x + dx, self.y + dy)

    def as_tuple(self) -> tuple[int, int]:
        """Return as (x, y) tuple."""
        return (self.x, self.y)


# Common direction vectors
NORTH = Point(0, -1)
SOUTH = Point(0, 1)
EAST = Point(1, 0)
WEST = Point(-1, 0)
DIRECTIONS_4 = [NORTH, EAST, SOUTH, WEST]
DIRECTIONS_8 = [Point(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx or dy]


class Grid[T]:
    """A 2D grid with convenient accessors."""

    def __init__(self, data: list[list[T]]):
        self.data = data
        self.height = len(data)
        self.width = len(data[0]) if data else 0

    @classmethod
    def from_string(cls, s: str, transform=None) -> Self:
        """Create a grid from a string."""
        lines = s.strip().splitlines()
        if transform:
            data = [[transform(c) for c in line] for line in lines]
        else:
            data = [list(line) for line in lines]
        return cls(data)

    def __getitem__(self, pos: Point | tuple[int, int]) -> T:
        if isinstance(pos, Point):
            return self.data[pos.y][pos.x]
        x, y = pos
        return self.data[y][x]

    def __setitem__(self, pos: Point | tuple[int, int], value: T):
        if isinstance(pos, Point):
            self.data[pos.y][pos.x] = value
        else:
            x, y = pos
            self.data[y][x] = value

    def __contains__(self, pos: Point | tuple[int, int]) -> bool:
        if isinstance(pos, Point):
            return 0 <= pos.x < self.width and 0 <= pos.y < self.height
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def __iter__(self) -> Iterator[tuple[Point, T]]:
        """Iterate over (point, value) pairs."""
        for y in range(self.height):
            for x in range(self.width):
                yield Point(x, y), self.data[y][x]

    def find(self, value: T) -> Point | None:
        """Find the first occurrence of a value."""
        for pos, v in self:
            if v == value:
                return pos
        return None

    def find_all(self, value: T) -> list[Point]:
        """Find all occurrences of a value."""
        return [pos for pos, v in self if v == value]

    def neighbors4(self, pos: Point) -> Iterator[tuple[Point, T]]:
        """Yield valid 4-connected neighbors."""
        for n in pos.neighbors4():
            if n in self:
                yield n, self[n]

    def neighbors8(self, pos: Point) -> Iterator[tuple[Point, T]]:
        """Yield valid 8-connected neighbors."""
        for n in pos.neighbors8():
            if n in self:
                yield n, self[n]

    def copy(self) -> Self:
        """Return a deep copy of this grid."""
        return type(self)(copy.deepcopy(self.data))
