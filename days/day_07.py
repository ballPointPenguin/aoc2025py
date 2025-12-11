import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 7

        [Puzzle Link](https://adventofcode.com/2025/day/7)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    # from more_itertools import chunked
    import sys
    from collections import defaultdict

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import EAST, SOUTH, WEST, Grid, get_puzzle
    return EAST, Grid, SOUTH, WEST, defaultdict, get_puzzle


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 7
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
def _(Grid, raw_input):
    def parse_input(data):
        """Parse manifold diagram."""
        grid = Grid.from_string(data)
        return grid

    # Parse input for Part 1
    grid = parse_input(raw_input)
    return grid, parse_input


@app.cell
def _(mo, parse_input, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_grid = parse_input(test_ex.input_data)
            print(f"\n{'=' * 60}")
            print(f"EXAMPLE {test_i + 1} DEBUG:")
            print(f"{'=' * 60}")
            result = solve_part1(example_grid, debug=True)
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
def _(EAST, SOUTH, WEST, grid):
    # Solve Part 1
    def solve_part1(grid_1, debug=False):
        """Solve part 1 of the puzzle."""
        start = grid_1.find("S")  # Find start in the grid we're solving
        beams = [start]
        split_count = 0
        step = 0

        if debug:
            print(f"Starting at: {start}")
            print(f"Grid size: {grid_1.width}x{grid_1.height}")

        while beams:
            step += 1
            if debug:
                print(f"\n=== Step {step} ===")
                print(f"Active beams: {len(beams)}")
                print(f"Beam positions: {beams}")

            new_beams = []
            for pos in beams:
                next_pos = pos + SOUTH

                if debug:
                    print(f"  Beam at {pos} moving to {next_pos}")

                if next_pos not in grid_1:
                    if debug:
                        print("    -> Out of bounds")
                    continue

                cell = grid_1[next_pos]
                if debug:
                    print(f"    -> Cell: '{cell}'")

                if cell == "^":
                    split_count += 1
                    left = next_pos + WEST
                    right = next_pos + EAST
                    new_beams.append(left)
                    new_beams.append(right)
                    if debug:
                        print(
                            f"    -> SPLIT! Count: {split_count}, new beams at {left} and {right}"
                        )
                else:
                    new_beams.append(next_pos)
                    if debug:
                        print(f"    -> Continue at {next_pos}")

            beams = list(set(new_beams))  # Deduplicate overlapping beams
            if debug:
                print(f"Next step will have {len(beams)} beams")

        if debug:
            print("\n=== DONE ===")
            print(f"Total splits: {split_count}")

        return split_count

    answer1 = solve_part1(grid)
    print(f"Part 1: {answer1}")
    return (solve_part1,)


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
                _example_grid_p2 = parse_input(_ex.input_data)
                _result_p2 = solve_part2(_example_grid_p2)
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
def _(EAST, SOUTH, WEST, defaultdict, grid):
    # Solve Part 2
    def solve_part2(grid_2):
        """Solve part 2 of the puzzle."""
        start = grid_2.find("S")
        beam_counts = {start: 1}  # position -> no. of timelines there
        exit_count = 0

        while beam_counts:
            new_counts = defaultdict(int)

            for pos, count in beam_counts.items():
                next_pos = pos + SOUTH

                if next_pos not in grid_2:
                    exit_count += count
                    continue

                cell = grid_2[next_pos]

                if cell == "^":
                    # Split timelines
                    new_counts[next_pos + WEST] += count
                    new_counts[next_pos + EAST] += count
                else:
                    new_counts[next_pos] += count

            beam_counts = dict(new_counts)

        return exit_count

    answer2 = solve_part2(grid)
    print(f"Part 2: {answer2}")
    return (solve_part2,)


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    _Add your notes, observations, and approach explanations here._
    """)
    return


if __name__ == "__main__":
    app.run()
