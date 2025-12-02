"""Common parsing utilities for Advent of Code puzzles."""

import re
from collections.abc import Callable


def parse_lines(data: str, strip: bool = True) -> list[str]:
    """Parse input into lines.

    Args:
        data: Raw input string
        strip: Whether to strip whitespace from each line

    Returns:
        List of lines
    """
    lines = data.splitlines()
    if strip:
        lines = [line.strip() for line in lines]
    return lines


def parse_ints(data: str) -> list[int]:
    """Extract all integers from a string (including negative).

    Args:
        data: Input string containing integers

    Returns:
        List of all integers found
    """
    return [int(x) for x in re.findall(r"-?\d+", data)]


def parse_grid[T](data: str, transform: Callable[[str], T] | None = None) -> list[list[T]]:
    """Parse input into a 2D grid.

    Args:
        data: Input string with newline-separated rows
        transform: Optional function to transform each cell (default: identity)

    Returns:
        2D list of grid cells
    """
    lines = parse_lines(data)
    if transform is None:
        return [list(line) for line in lines]
    return [[transform(c) for c in line] for line in lines]


def parse_sections(data: str) -> list[str]:
    """Split input into sections separated by blank lines.

    Args:
        data: Input string with sections separated by blank lines

    Returns:
        List of section strings
    """
    return data.strip().split("\n\n")


def parse_grid_dict[T](
    data: str, transform: Callable[[str], T] | None = None
) -> dict[tuple[int, int], T]:
    """Parse input into a dict mapping (row, col) -> value.

    Useful for sparse grids or when you need coordinate access.

    Args:
        data: Input string with newline-separated rows
        transform: Optional function to transform each cell

    Returns:
        Dict mapping (row, col) coordinates to values
    """
    grid = {}
    for row, line in enumerate(parse_lines(data)):
        for col, char in enumerate(line):
            value = transform(char) if transform else char
            grid[(row, col)] = value
    return grid
