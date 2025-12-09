import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 5

        [Puzzle Link](https://adventofcode.com/2025/day/5)
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
    from aoc_utils import get_puzzle  # , parse_lines, parse_ints
    return (get_puzzle,)


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 5
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
    # Parse the two-section input
    def parse_input(data):
        """Parse input into ranges and IDs.

        Returns:
            ranges: List of (start, end) tuples representing inclusive ranges
            ids: List of ingredient IDs to check
        """
        # Split on blank line to separate sections
        sections = data.strip().split("\n\n")
        range_section, id_section = sections[0], sections[1]

        # Parse ranges: "3-5" -> (3, 5)
        ranges = []
        for line in range_section.strip().split("\n"):
            start, end = line.split("-")
            ranges.append((int(start), int(end)))

        # Parse IDs: one per line
        ids = [int(line) for line in id_section.strip().split("\n")]

        return ranges, ids

    # Parse input for Part 1
    ranges, ids = parse_input(raw_input)

    # Preview
    print(f"Ranges: {ranges[:5]}...")
    print(f"IDs: {ids[:10]}...")
    print(f"Total ranges: {len(ranges)}, Total IDs: {len(ids)}")
    return ids, parse_input, ranges


@app.cell
def _(mo, parse_input, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_ranges, example_ids = parse_input(test_ex.input_data)
            result = solve_part1(example_ranges, example_ids)
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
def _(ids, ranges):
    # Solve Part 1
    def solve_part1(ranges, ids):
        """Count how many IDs fall within any range.

        Args:
            ranges: List of (start, end) tuples (inclusive)
            ids: List of ingredient IDs to check

        Returns:
            Count of fresh ingredient IDs
        """
        fresh_items = 0

        for id in ids:
            for range in ranges:
                start, end = range
                if start <= id <= end:
                    fresh_items += 1
                    break

        return fresh_items

    answer1 = solve_part1(ranges, ids)
    print(f"Part 1: {answer1}")
    return (solve_part1,)


@app.cell
def _(ids, ranges):
    # Solve Part 1 with JAX (vectorized)
    import jax
    import jax.numpy as jnp

    # Enable 64-bit precision in JAX (needed for huge ingredient IDs)
    jax.config.update("jax_enable_x64", True)

    def solve_part1_jax(ranges, ids):
        """Vectorized solution using JAX broadcasting.

        Creates a (n_ids × n_ranges) boolean matrix where each cell
        indicates if that ID falls within that range.
        """
        # Convert to JAX arrays with int64 (ingredient IDs are huge!)
        ids_array = jnp.array(ids, dtype=jnp.int64)  # shape: (1000,)
        starts = jnp.array([r[0] for r in ranges], dtype=jnp.int64)  # shape: (180,)
        ends = jnp.array([r[1] for r in ranges], dtype=jnp.int64)  # shape: (180,)

        # Reshape ids for broadcasting: (1000,) → (1000, 1)
        ids_col = ids_array[:, jnp.newaxis]

        # Broadcasting magic:
        # ids_col shape: (1000, 1)
        # starts shape:  (180,)
        # Result shape:  (1000, 180) - all comparisons at once!
        in_range = (ids_col >= starts) & (ids_col <= ends)

        # For each ID (row), check if it matches ANY range (column)
        # any(axis=1) reduces (1000, 180) → (1000,)
        matches_any_range = jnp.any(in_range, axis=1)

        # Count how many IDs match at least one range
        fresh_count = jnp.sum(matches_any_range)

        return int(fresh_count)

    answer1_jax = solve_part1_jax(ranges, ids)
    print(f"Part 1 JAX: {answer1_jax}")
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
                _example_ranges_p2, _example_ids_p2 = parse_input(_ex.input_data)
                _result_p2 = solve_part2(_example_ranges_p2)
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
def _(ranges):
    # Solve Part 2
    def solve_part2(ranges):
        """Count unique IDs across all ranges by merging overlaps."""
        if not ranges:
            return 0

        # Sort ranges by start position
        sorted_ranges = sorted(ranges, key=lambda r: r[0])

        # Merge overlapping/adjacent ranges
        merged = []
        current_start, current_end = sorted_ranges[0]

        for start, end in sorted_ranges[1:]:
            # Check if current range overlaps or is adjacent to the next
            if start <= current_end + 1:
                # Merge: extend current_end if needed
                current_end = max(current_end, end)
            else:
                # No overlap: save current range and start a new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Don't forget the last range
        merged.append((current_start, current_end))

        # Count total IDs in merged ranges
        total_fresh = sum(end - start + 1 for start, end in merged)

        return total_fresh

    answer2 = solve_part2(ranges)
    print(f"Part 2: {answer2}")
    return (solve_part2,)


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    Part 1 "naïve" solution runs in just 5ms.

    Part 1 with JAX runs in 134ms.

    27x slower with JAX due to:

        - no early exit
        - requires 64-bit arithmetic
        - array creation overhead
    """)
    return


if __name__ == "__main__":
    app.run()
