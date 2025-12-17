import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 10

        [Puzzle Link](https://adventofcode.com/2025/day/1)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    import sys
    from collections import deque

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import get_puzzle  # , parse_lines, parse_ints

    return Bounds, LinearConstraint, coo_matrix, deque, get_puzzle, milp, np


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 10
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
    import re

    def parse_line(line):
        # Extract lights: [.##.]
        # 0 means first light.
        lights_match = re.search(r"\[([.#]+)\]", line)
        if not lights_match:
            return None
        lights_str = lights_match.group(1)

        # Binary representation of target state
        # 0-indexed, corresponding to light position
        target_bin = 0
        for i, char in enumerate(lights_str):
            if char == "#":
                target_bin |= 1 << i

        # Extract buttons: (3) (1,3) ...
        # Find all content within parentheses
        button_strs = re.findall(r"\(([^)]+)\)", line)
        buttons = []
        buttons_bin = []

        for b_str in button_strs:
            # Split by comma, strip whitespace, convert to int
            indices = tuple(int(x.strip()) for x in b_str.split(",") if x.strip())
            buttons.append(indices)

            # Binary mask for button
            mask = 0
            for idx in indices:
                mask |= 1 << idx
            buttons_bin.append(mask)

        # Extract joltage: {3,5,4,7}
        joltage_match = re.search(r"\{([^}]+)\}", line)
        joltage = []
        if joltage_match:
            joltage = [int(x.strip()) for x in joltage_match.group(1).split(",") if x.strip()]

        return {
            "target_str": lights_str,
            "target_bin": target_bin,
            "num_lights": len(lights_str),
            "buttons": buttons,
            "buttons_bin": buttons_bin,
            "joltage": joltage,
        }

    def parse_input(data):
        """Parse input into structured data."""
        lines = data.strip().split("\n")
        parsed = []
        for line in lines:
            if not line.strip():
                continue
            p = parse_line(line)
            if p:
                parsed.append(p)
        return parsed

    # Parse input for Part 1
    lines = parse_input(raw_input)
    print(f"Example Input: {lines[0]}")
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
def _(deque, lines):
    # Solve Part 1
    def solve_machine(machine):
        target = machine["target_bin"]
        buttons = machine["buttons_bin"]

        # Optimization: If target is 0, 0 presses needed
        if target == 0:
            return 0

        # Use Double-Ended Queue "deque" for BFS
        queue = deque([(0, 0)])  # (current_state, presses)
        seen = {0}  # bitmask of states visited

        while queue:
            # Remove the item that has been waiting longest (FIFO)
            # This is O(1) for a deque, but O(N) for a list
            current_state, presses = queue.popleft()

            # Try pressing each button
            for btn_mask in buttons:
                # XOR to toggle lights
                next_state = current_state ^ btn_mask

                if next_state == target:
                    return presses + 1

                if next_state not in seen:
                    seen.add(next_state)
                    # Add new states to the back
                    queue.append((next_state, presses + 1))

        return -1  # Should be unreachable

    def solve_part1(data):
        """Solve part 1 of the puzzle."""
        solutions = [solve_machine(machine) for machine in data]

        return sum(solutions)

    answer1 = solve_part1(lines)
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
def _(Bounds, LinearConstraint, coo_matrix, lines, milp, np):
    # Solve Part 2
    def solve_machine_p2(machine):
        b = np.asarray(machine["joltage"], dtype=int)  # shape (m,)
        buttons = machine["buttons"]  # length n
        m = b.size
        n = len(buttons)

        # Build sparse A (m x n) with 0/1 entries
        rows = []
        cols = []
        data = []
        for j, idxs in enumerate(buttons):
            for i in idxs:
                rows.append(i)
                cols.append(j)
                data.append(1)

        A = coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()

        # Objective: minimize sum x_j
        c = np.ones(n, dtype=float)

        # Equality constraints: A x == b
        constraints = LinearConstraint(A, lb=b, ub=b)

        # Bounds: x_j >= 0 (no real upper bound needed)
        bounds = Bounds(lb=np.zeros(n), ub=np.full(n, np.inf))

        # Integrality: all variables are integers
        integrality = np.ones(n, dtype=int)

        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

        if not res.success:
            raise RuntimeError(res.message)

        # Optimal objective value is the minimum total presses
        return int(round(res.fun))

    def solve_part2(data):
        """Solve part 2 of the puzzle."""
        solutions = [solve_machine_p2(machine) for machine in data]
        return sum(solutions)

    answer2 = solve_part2(lines)
    print(f"Part 2: {answer2}")
    return (solve_part2,)


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    Finding the shortest path in an unweighted graph via BFS.

    Part 1: system of linear equations over a Galois Field of 2 elements, GF(2)

    Part 2: Integer Linear Program (ILP)

    or MILP: minimum 1-norm nonnegative integer solution

    scipy.optimize.milp (backed by HiGHS) directly models:

    Ax = b, x ≥ 0, integer, minimize ∑ x
    """)
    return


if __name__ == "__main__":
    app.run()
