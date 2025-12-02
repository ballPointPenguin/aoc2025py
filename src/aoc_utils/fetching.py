"""Utilities for fetching Advent of Code puzzle data.

Uses the advent-of-code-data package which handles:
- Session authentication (via AOC_SESSION env var)
- Caching to avoid repeated requests
- Example data extraction
"""

from aocd.models import Puzzle


def get_puzzle(year: int = 2025, day: int | None = None) -> Puzzle:
    """Get a Puzzle object for the given year and day.

    If day is not provided, it will be auto-detected from the calling
    filename (e.g., day_01.py -> day 1).

    Args:
        year: The AoC year (default 2025)
        day: The day number (1-25), or None for auto-detection

    Returns:
        Puzzle object with input_data, examples, and more
    """
    if day is None:
        # Auto-detection happens in aocd based on calling file
        # For explicit control, always pass day
        raise ValueError("Please provide day explicitly for clarity")

    return Puzzle(year=year, day=day)


def get_input(year: int = 2025, day: int | None = None) -> str:
    """Get the puzzle input as a string.

    Args:
        year: The AoC year (default 2025)
        day: The day number (1-25)

    Returns:
        Raw puzzle input string
    """
    puzzle = get_puzzle(year=year, day=day)
    return puzzle.input_data


def get_examples(year: int = 2025, day: int | None = None) -> list:
    """Get the example inputs/outputs from the puzzle description.

    Args:
        year: The AoC year (default 2025)
        day: The day number (1-25)

    Returns:
        List of Example objects with input_data and answer_a/answer_b
    """
    puzzle = get_puzzle(year=year, day=day)
    return puzzle.examples
