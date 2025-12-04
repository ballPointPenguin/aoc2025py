import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 2

        [Puzzle Link](https://adventofcode.com/2025/day/2)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    # from more_itertools import chunked
    import sys

    import polars as pl

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import get_puzzle  # , parse_lines, parse_ints
    return get_puzzle, pl


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 2
    puzzle = get_puzzle(year=2025, day=DAY)
    raw_input = puzzle.input_data
    return puzzle, raw_input


@app.cell
def _(mo, puzzle):
    # Show examples if available
    examples = puzzle.examples
    if examples:
        example_parts = []
        for i, ex in enumerate(examples):
            part_text = f"**Example {i + 1}:**\n```\n{ex.input_data}\n```\n"
            part_text += f"Expected Part 1: `{ex.answer_a}`"
            if ex.answer_b:
                part_text += f"\n\nExpected Part 2: `{ex.answer_b}`"
            example_parts.append(part_text)
        example_text = "\n\n".join(example_parts)
        # Return the markdown object to ensure it is displayed
        display_ex = mo.md(f"## Examples\n\n{example_text}")
    else:
        display_ex = mo.md("_No examples parsed from puzzle description._")

    display_ex
    return


@app.cell
def _(mo, raw_input):
    # Preview the input
    preview = raw_input[:500] + "..." if len(raw_input) > 500 else raw_input
    mo.md(f"## Input Preview\n```\n{preview}\n```")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1
    """)
    return


@app.cell
def _(raw_input):
    # TODO: Customize parsing based on actual puzzle
    # Generic parsing function
    def parse_input(data):
        """Parse input as lines."""
        return data.strip().split("\n")

    # Parse input for Part 1
    lines = parse_input(raw_input)
    lines[:5]  # Preview first 5 lines
    return lines, parse_input


@app.cell
def _(mo, parse_input, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_lines = parse_input(test_ex.input_data)
            result = solve_part1(example_lines)
            expected = test_ex.answer_a
            match = "✓" if result == int(expected) else "✗"
            example_results.append(
                f"{match} Example {test_i + 1}: got {result}, expected {expected}"
            )
        display_test = mo.md("## Example Validation\n\n" + "\n\n".join(example_results))
    else:
        display_test = mo.md("_No examples parsed from puzzle description._")

    display_test
    return


@app.cell
def _(lines):
    # Solve Part 1
    def solve_part1(data):
        """Solve part 1 of the puzzle."""
        # TODO: Implement solution
        return None

    answer1 = solve_part1(lines)
    print(f"Part 1: {answer1}")
    return (solve_part1,)


@app.cell
def _(lines, pl):
    # Solve Part 1 with Polars
    def solve_part1_pl(data):
        """Solve part 1 using Polars - the over-engineered version!"""

        # Parse input into a DataFrame
        df = pl.DataFrame({"instruction": data})

        # TODO solve part 1 with pl

        return None, df

    answer1_pl, df = solve_part1_pl(lines)
    print(f"Part 1: {answer1_pl}")
    print(f"Part 1 DataFrame: {df}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2
    """)
    return


@app.cell
def _(mo, parse_input, puzzle, solve_part2):
    # Test against examples if available for Part 2
    if puzzle.examples:
        _example_results_p2 = []
        for _i, _ex in enumerate(puzzle.examples):
            if _ex.answer_b:  # Only test if Part 2 answer exists
                _example_lines_p2 = parse_input(_ex.input_data)
                _result_p2 = solve_part2(_example_lines_p2)
                _expected_p2 = _ex.answer_b
                _match_p2 = "✓" if _result_p2 == int(_expected_p2) else "✗"
                _example_results_p2.append(
                    f"{_match_p2} Example {_i + 1}: got {_result_p2}, expected {_expected_p2}"
                )
        if _example_results_p2:
            display_test_p2 = mo.md(
                "## Part 2 Example Validation\n\n" + "\n\n".join(_example_results_p2)
            )
        else:
            display_test_p2 = mo.md("_Part 2 examples not yet available._")
    else:
        display_test_p2 = mo.md("_No examples parsed from puzzle description._")

    display_test_p2
    return


@app.cell
def _(lines):
    # Solve Part 2
    def solve_part2(data):
        """Solve part 2 of the puzzle."""
        # TODO: Implement solution
        return None

    answer2 = solve_part2(lines)
    print(f"Part 2: {answer2}")
    return (solve_part2,)


@app.cell
def _(lines, pl):
    # Solve Part 2 with Polars
    def solve_part2_pl(data):
        # Parse input into a DataFrame
        df_2 = pl.DataFrame({"instruction": data})

        # TODO solve part 2 with pl

        return None, df_2

    answer2_pl, df_2 = solve_part2_pl(lines)
    print(f"Part 2: {answer2_pl}")
    print(f"Part 2 DataFrame: {df_2}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    _Add your notes, observations, and approach explanations here._
    """)
    return


if __name__ == "__main__":
    app.run()
