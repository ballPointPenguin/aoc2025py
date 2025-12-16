import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 9

        [Puzzle Link](https://adventofcode.com/2025/day/1)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    import sys
    from itertools import combinations

    import numpy as np

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import get_puzzle  # , parse_lines, parse_ints

    return combinations, get_puzzle, np


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 9
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
    input_len = len(raw_input)
    preview = raw_input[:500] + "..." if input_len > 500 else raw_input
    mo.md(f"## Input Preview\nlength: {input_len}\n```\n{preview}\n```")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1
    """)
    return


@app.cell
def _(raw_input):
    def parse_input(data):
        """Parse input as tuples (x,y)"""
        lines = data.strip().split("\n")
        return [tuple(map(int, line.split(","))) for line in lines]

    # Parse input for Part 1
    tiles = parse_input(raw_input)
    tiles[:5]  # Preview first 5 lines
    return parse_input, tiles


@app.cell
def _(mo, parse_input, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_lines = parse_input(test_ex.input_data)
            result = solve_part1(example_lines)
            # expected = test_ex.answer_a
            expected = 50
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
def _(combinations, tiles):
    # Solve Part 1

    def find_area(combo):
        (x1, y1), (x2, y2) = combo
        return abs(x1 - x2 + 1) * abs(y1 - y2 + 1)

    def solve_part1(data):
        """Solve part 1 of the puzzle."""
        combos = combinations(data, 2)
        areas = [find_area(combo) for combo in combos]

        return max(areas)

    answer1 = solve_part1(tiles)
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
    if puzzle.examples:
        _example_results_p2 = []
        for _i, _ex in enumerate(puzzle.examples):
            if _ex.answer_b:  # Only test if Part 2 answer exists
                _example_lines_p2 = parse_input(_ex.input_data)
                _result_p2 = solve_part2(_example_lines_p2)
                _expected_p2 = 24
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
def _(np, tiles):
    def build_integral_image(tiles):
        # 1. Coordinate Compression
        xs = sorted(set(t[0] for t in tiles))
        ys = sorted(set(t[1] for t in tiles))

        # Map real coords -> grid indices
        x_map = {x: i for i, x in enumerate(xs)}
        y_map = {y: i for i, y in enumerate(ys)}

        # Grid dimensions (number of elementary intervals)
        # Grid cells [i, j] represent the rectangular region
        # between x[i]...x[i+1] and y[j]...y[j+1].
        # So grid size is (len(xs)-1, len(ys)-1).
        H = len(ys) - 1
        W = len(xs) - 1
        grid = np.zeros((H, W), dtype=bool)

        # 2. Draw Polygon Boundary
        # Collect vertical edges
        vert_edges = []
        num_tiles = len(tiles)
        for k in range(num_tiles):
            p1 = tiles[k]
            p2 = tiles[(k + 1) % num_tiles]

            # If vertical edge
            if p1[0] == p2[0]:
                x = p1[0]
                y_min, y_max = sorted((p1[1], p2[1]))
                # Map to compressed indices
                xi = x_map[x]
                yi_min = y_map[y_min]
                yi_max = y_map[y_max]

                # Record that at x-index `xi`,
                # we have a toggle for y-ranges `yi_min` to `yi_max`
                vert_edges.append((xi, yi_min, yi_max))

        # Fill the grid
        for j in range(H):  # For each y-interval
            # Find vertical edges that span this y-interval
            row_crossings = []
            for xi, y_start, y_end in vert_edges:
                if y_start <= j < y_end:
                    row_crossings.append(xi)

            row_crossings.sort()

            # Fill between pairs: indices 0-1, 2-3, etc are INSIDE
            for k in range(0, len(row_crossings), 2):
                x_start = row_crossings[k]
                x_end = row_crossings[k + 1]
                grid[j, x_start:x_end] = True

        # 3. Build Integral Image (2D Prefix Sum)
        # This allows O(1) area lookups in "grid cell counts"
        # Integral image `S[y, x]` = sum of `grid[:y, :x]`
        # We pad with one row/col of zeros for easier indexing
        integral = np.zeros((H + 1, W + 1), dtype=int)
        integral[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)

        return integral, x_map, y_map

    def get_validity_mask(integral, xi1, yi1, xi2, yi2):
        """Vectorized validity check using integral image."""
        # Ensure coordinates are min/max for the integral query
        xi_min = np.minimum(xi1, xi2)
        xi_max = np.maximum(xi1, xi2)
        yi_min = np.minimum(yi1, yi2)
        yi_max = np.maximum(yi1, yi2)

        expected_counts = (xi_max - xi_min) * (yi_max - yi_min)

        # Vectorized lookup in integral image
        # S[y2, x2] - S[y1, x2] - S[y2, x1] + S[y1, x1]
        term1 = integral[yi_max, xi_max]
        term2 = integral[yi_min, xi_max]
        term3 = integral[yi_max, xi_min]
        term4 = integral[yi_min, xi_min]

        actual_counts = term1 - term2 - term3 + term4

        return actual_counts == expected_counts

    # Solve Part 2
    def solve_part2(tiles):
        """Solve part 2 of the puzzle."""
        integral, x_map, y_map = build_integral_image(tiles)

        # Convert tiles to arrays
        T = np.array(tiles)
        Xs = T[:, 0]
        Ys = T[:, 1]

        # We need all pairs (i, j) where i < j
        # Construct upper triangle indices for efficient pair generation
        i_idx, j_idx = np.triu_indices(len(tiles), k=1)

        # Extract coords for pairs
        # This copies data, creating ~18M length arrays
        x1 = Xs[i_idx]
        y1 = Ys[i_idx]
        x2 = Xs[j_idx]
        y2 = Ys[j_idx]

        # Calculate areas (inclusive area formula from Part 1)
        areas = (np.abs(x1 - x2) + 1) * (np.abs(y1 - y2) + 1)

        # Map coords to grid indices ONCE for all tiles to speed up the check
        mapped_xs = np.array([x_map[x] for x in Xs])
        mapped_ys = np.array([y_map[y] for y in Ys])

        # Get the mapped coords for the pairs
        xi1 = mapped_xs[i_idx]
        yi1 = mapped_ys[i_idx]
        xi2 = mapped_xs[j_idx]
        yi2 = mapped_ys[j_idx]

        # Vectorized check if rectangles are fully inside polygon
        is_valid = get_validity_mask(integral, xi1, yi1, xi2, yi2)

        # Filter for valid areas and find max
        valid_areas = areas[is_valid]

        if len(valid_areas) == 0:
            return 0

        return np.max(valid_areas)

    answer2 = solve_part2(tiles)
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
