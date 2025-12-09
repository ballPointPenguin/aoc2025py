import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 4

        [Puzzle Link](https://adventofcode.com/2025/day/4)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    # from more_itertools import chunked
    import sys

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import Grid, get_puzzle  # , parse_lines, parse_ints

    return Grid, get_puzzle


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 4
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
        """Parse into a Grid"""
        return Grid.from_string(data)

    grid = parse_input(raw_input)
    return grid, parse_input


@app.cell
def _(mo, parse_input, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_grid = parse_input(test_ex.input_data)
            result = solve_part1(example_grid)
            # expected = test_ex.answer_a
            expected = 13
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
def _(grid):
    # Solve Part 1
    def solve_part1(data):
        """Solve part 1 of the puzzle."""
        rolls = data.find_all("@")
        accessible = 0

        for pos in rolls:
            neighbor_count = sum(1 for _, val in data.neighbors8(pos) if val == "@")
            if neighbor_count < 4:
                accessible += 1

        return accessible

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
def _(grid):
    # Solve Part 2
    def find_accessible_rolls(grid):
        rolls = grid.find_all("@")
        accessible = []

        for pos in rolls:
            neighbor_count = sum(1 for _, val in grid.neighbors8(pos) if val == "@")
            if neighbor_count < 4:
                accessible.append(pos)

        return accessible

    def solve_part2(grid):
        """Solve part 2 of the puzzle."""
        # Work on a copy to avoid mutating the original
        grid = grid.copy()
        total_removed = 0

        while True:
            accessible = find_accessible_rolls(grid)
            if not accessible:
                break
            for pos in accessible:
                grid[pos] = "."
            total_removed += len(accessible)

        return total_removed

    answer2 = solve_part2(grid)
    print(f"Part 2: {answer2}")
    return (solve_part2,)


@app.cell
def _(raw_input):
    # Solve Part 2 with JAX (vectorized)
    # NOTE: We parse fresh from raw_input because the previous cell mutated `grid`
    import jax
    import jax.numpy as jnp
    from jax.scipy.signal import convolve

    def solve_part2_jax(raw_data, debug=False):
        """Solve Part 2 using JAX vectorized operations."""
        # Parse fresh from raw input (avoid mutation issues)
        lines = raw_data.strip().split("\n")

        # Convert grid to binary array: @ = 1, . = 0
        grid_array = jnp.array(
            [[1 if cell == "@" else 0 for cell in line] for line in lines],
            dtype=jnp.int32,
        )

        if debug:
            print(f"Grid shape: {grid_array.shape}")
            print(f"Initial @ count: {int(jnp.sum(grid_array))}")

        # 3x3 kernel for counting 8 neighbors (center is 0)
        kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.int32)

        # Debug: test convolution on initial grid
        if debug:
            test_counts = convolve(grid_array, kernel, mode="same", method="direct")
            accessible = (grid_array == 1) & (test_counts < 4)
            print(f"Initial accessible count: {int(jnp.sum(accessible))}")

        def body_fun(carry):
            """One iteration: find accessible, remove them, count."""
            current_grid, total_removed = carry

            # Count neighbors for each cell via convolution
            neighbor_counts = convolve(current_grid, kernel, mode="same", method="direct")

            # Find accessible: has paper roll (1) AND < 4 neighbors
            accessible_mask = (current_grid == 1) & (neighbor_counts < 4)

            # Count how many we're removing this round
            num_removed = jnp.sum(accessible_mask)

            # Remove them (set to 0)
            new_grid = jnp.where(accessible_mask, 0, current_grid)

            return (new_grid, total_removed + num_removed)

        def cond_fun(carry):
            """Continue while there are accessible rolls."""
            current_grid, _ = carry
            neighbor_counts = convolve(current_grid, kernel, mode="same", method="direct")
            accessible_mask = (current_grid == 1) & (neighbor_counts < 4)
            return jnp.sum(accessible_mask) > 0

        # Run the functional while loop
        init_val = (grid_array, 0)
        _final_grid, total_removed = jax.lax.while_loop(cond_fun, body_fun, init_val)

        if debug:
            print(f"Final @ count: {int(jnp.sum(_final_grid))}")

        return int(total_removed)

    answer2_jax = solve_part2_jax(raw_input, debug=False)
    print(f"Part 2 w/ JAX: {answer2_jax}")
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
