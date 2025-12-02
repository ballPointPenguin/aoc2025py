import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - TEMPLATE DAY

        [Puzzle Link](https://adventofcode.com/2025/day/1)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    # import polars as pl
    # from more_itertools import chunked
    import sys

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import get_puzzle  # , parse_lines, parse_ints

    return (get_puzzle,)


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 0
    puzzle = get_puzzle(year=2025, day=DAY)
    raw_input = puzzle.input_data
    return puzzle, raw_input


@app.cell
def _(mo, puzzle):
    # Show examples if available
    examples = puzzle.examples
    if examples:
        example_text = "\n\n".join(
            [
                f"**Example {i + 1}:**\n```\n{ex.input_data}\n```\nExpected Part A: `{ex.answer_a}`"
                for i, ex in enumerate(examples)
            ]
        )
        # Return the markdown object to ensure it is displayed
        display = mo.md(f"## Examples\n\n{example_text}")
    else:
        display = mo.md("_No examples parsed from puzzle description._")

    display


@app.cell
def _(mo, raw_input):
    # Preview the input
    preview = raw_input[:500] + "..." if len(raw_input) > 500 else raw_input
    mo.md(f"## Input Preview\n```\n{preview}\n```")


@app.cell
def _(mo):
    mo.md(
        """
    ## Part 1
    """
    )


@app.cell
def _(raw_input):
    # Parse input for Part 1
    # TODO: Customize parsing based on actual puzzle

    # Example: parse as lines
    lines = raw_input.strip().split("\n")

    # Example: parse all integers from input
    # numbers = parse_ints(raw_input)

    lines[:5]  # Preview first 5 lines
    return (lines,)


@app.cell
def _(lines):
    # Solve Part 1
    def solve_part1(data):
        """Solve part 1 of the puzzle."""
        # TODO: Implement solution
        return None

    answer1 = solve_part1(lines)
    print(f"Part 1: {answer1}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Part 2
    """
    )


@app.cell
def _(lines):
    # Solve Part 2
    def solve_part2(data):
        """Solve part 2 of the puzzle."""
        # TODO: Implement solution
        return None

    answer2 = solve_part2(lines)
    print(f"Part 2: {answer2}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Notes

    _Add your notes, observations, and approach explanations here._
    """
    )


if __name__ == "__main__":
    app.run()
