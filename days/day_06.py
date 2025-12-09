import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 6

        [Puzzle Link](https://adventofcode.com/2025/day/6)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    # from more_itertools import chunked
    import sys
    from math import prod

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import get_puzzle  # , parse_lines, parse_ints

    return get_puzzle, prod


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 6
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
    # Parse the "horizontal" math worksheet into vertical problems
    def parse_input_1(data):
        """Parse math worksheet into vertical problems.

        Returns:
            problems: List of lists, where each inner list is the numbers in a problem
            operators: List of operators ('*' or '+') for each problem
        """
        lines = data.strip().split("\n")

        # Parse all rows except last: split on whitespace and convert to ints
        number_rows = []
        for line in lines[:-1]:
            numbers = [int(x) for x in line.split()]
            number_rows.append(numbers)

        # Transpose rows → columns using zip(*rows)
        # This rotates [[1,2,3], [4,5,6]] into [(1,4), (2,5), (3,6)]
        problems = [list(col) for col in zip(*number_rows)]

        # Parse operators from last row
        operators = lines[-1].split()

        return problems, operators

    # Parse input for Part 1
    problems, operators = parse_input_1(raw_input)

    # Preview
    print(f"Found {len(problems)} problems")
    for idx in range(min(4, len(problems))):
        nums_str = " ".join(str(n) for n in problems[idx])
        print(f"Problem {idx + 1}: {nums_str} (operator: {operators[idx]})")
    return operators, parse_input_1, problems


@app.cell
def _(raw_input):
    # Parse for Part 2: right-to-left digit reading
    def parse_input_2(data):
        """Parse math worksheet reading digits right-to-left, preserving spacing.

        In Part 2, we read digit columns from right-to-left.
        The whitespace within each problem is significant for digit alignment.

        Returns:
            problems: List of lists, where each inner list is the new numbers
            operators: List of operators ('*' or '+') for each problem
        """
        lines = data.strip().split("\n")

        # Identify separator columns (all spaces across all rows)
        max_len = max(len(line) for line in lines)
        separator_cols = []
        for col in range(max_len):
            if all(col >= len(line) or line[col] == " " for line in lines):
                separator_cols.append(col)

        # Find problem regions (consecutive non-separator columns)
        regions = []
        start = None
        for col in range(max_len):
            if col not in separator_cols:
                if start is None:
                    start = col
            else:
                if start is not None:
                    regions.append((start, col))
                    start = None
        if start is not None:
            regions.append((start, max_len))

        # Extract number strings for each problem region (preserving spaces)
        problem_strings = []
        for start_col, end_col in regions:
            strings = []
            for line in lines[:-1]:  # All but operator row
                substr = line[start_col:end_col] if end_col <= len(line) else line[start_col:]
                strings.append(substr)
            problem_strings.append(strings)

        # Parse operators
        operators = lines[-1].split()

        # Now for each problem, read digits right-to-left
        part2_problems = []
        for str_list in problem_strings:
            # Find max width of this problem
            max_width = max(len(s) for s in str_list)

            # Read each digit position from right to left
            new_numbers = []
            for digit_pos in range(max_width - 1, -1, -1):  # Right to left
                digits = []
                for row_str in str_list:
                    if digit_pos < len(row_str) and row_str[digit_pos] != " ":
                        digits.append(row_str[digit_pos])

                if digits:
                    new_numbers.append(int("".join(digits)))

            part2_problems.append(new_numbers)

        return part2_problems, operators

    # Parse for Part 2
    problems_p2, operators_p2 = parse_input_2(raw_input)

    # Preview
    print(f"Part 2: Found {len(problems_p2)} problems")
    for idx_2 in range(min(4, len(problems_p2))):
        nums_str_2 = " ".join(str(n) for n in problems_p2[idx_2])
        print(f"Problem {idx_2 + 1}: {nums_str_2} (operator: {operators_p2[idx_2]})")
    return parse_input_2, problems_p2


@app.cell
def _(mo, parse_input_1, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_problems, example_operators = parse_input_1(test_ex.input_data)
            result = solve_part1(example_problems, example_operators)
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
def _(operators, problems, prod):
    # Solve Part 1
    def solve_part1(problems, operators):
        """Solve part 1 of the puzzle.

        Args:
            problems: List of lists, where each inner list is numbers for a problem
            operators: List of operators ('*' or '+') corresponding to each problem

        Returns:
            Grand total of all problem answers
        """
        total = 0
        for prob_i, numbers in enumerate(problems):
            if operators[prob_i] == "*":
                total += prod(numbers)
            else:
                total += sum(numbers)

        return total

    answer1 = solve_part1(problems, operators)
    print(f"Part 1: {answer1}")
    return (solve_part1,)


@app.cell
def _(mo):
    mo.md("""
    ## Part 2
    """)
    return


@app.cell
def _(mo, parse_input_2, puzzle, solve_part2):
    # Test against examples if available for Part 2
    if puzzle.examples:
        _example_results_p2 = []
        for _i, _ex in enumerate(puzzle.examples):
            if _ex.answer_b:  # Only test if Part 2 answer exists
                _example_problems_p2, _example_operators_p2 = parse_input_2(_ex.input_data)
                print(f"Example Problems Pt.2: {_example_problems_p2}")
                _result_p2 = solve_part2(_example_problems_p2, _example_operators_p2)
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
def _(operators, problems_p2, prod):
    # Solve Part 2
    def solve_part2(problems_p2, operators):
        """Solve part 2 of the puzzle.

        Args:
            problems: List of lists, where each inner list is numbers for a problem
            operators: List of operators ('*' or '+') corresponding to each problem

        Returns:
            Grand total of all problem answers
        """
        total = 0
        for prob_i, numbers in enumerate(problems_p2):
            if operators[prob_i] == "*":
                total += prod(numbers)
            else:
                total += sum(numbers)

        return total

    answer2 = solve_part2(problems_p2, operators)
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
