import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Advent of Code 2025 - Day 8

        [Puzzle Link](https://adventofcode.com/2025/day/8)
        """
    )
    return (mo,)


@app.cell
def _():
    # Imports
    # from more_itertools import chunked
    # import math
    import sys
    from itertools import combinations

    import numpy as np

    # Add src to path for local imports
    sys.path.insert(0, "../src")
    from aoc_utils import UnionFind, get_puzzle
    return UnionFind, combinations, get_puzzle, np


@app.cell
def _(get_puzzle):
    # Fetch puzzle data
    DAY = 8
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
    def parse_input(data):
        """Parse input as tuples (x,y,z)."""
        lines = data.strip().split("\n")
        return [tuple(map(int, line.split(","))) for line in lines]

    # Parse input for Part 1
    boxes = parse_input(raw_input)
    boxes[:5]  # Preview first 5 lines
    return boxes, parse_input


@app.cell
def _(mo, parse_input, puzzle, solve_part1):
    # Test against examples if available
    if puzzle.examples:
        example_results = []
        for test_i, test_ex in enumerate(puzzle.examples):
            example_boxes = parse_input(test_ex.input_data)
            result = solve_part1(example_boxes, 10)
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
def _(UnionFind, boxes, combinations):
    # Solve Part 1
    def calculate_distances(points):
        """Generate all pairwise distances (edges)"""
        edges = []
        for i, j in combinations(range(len(points)), 2):
            p1, p2 = points[i], points[j]
            # Use squared distances to avoid sqrt
            dist_sq = sum((a - b) ** 2 for a, b in zip(p1, p2))
            edges.append((dist_sq, i, j))

        print(f"Edges Found: {len(edges)}")
        return sorted(edges)

    def solve_part1(boxes, goal):
        """Solve part 1 of the puzzle."""
        edges = calculate_distances(boxes)

        # Init Union-Find for all points
        uf = UnionFind(len(boxes))

        for dist, i, j in edges[:goal]:
            uf.union(i, j)

        # Get sizes of all components
        sizes = uf.get_component_sizes()

        # Return product of 3 largest
        sizes.sort(reverse=True)
        result = sizes[0] * sizes[1] * sizes[2]

        return result

    answer1 = solve_part1(boxes, 1000)
    print(f"Part 1: {answer1}")
    return (solve_part1,)


@app.cell
def _(UnionFind, boxes, np):
    # Solve Part 1 with numpy
    def top_k_edges(points, k):
        # points: list[tuple[int,int,int]] -> (N,3) array
        P = np.asarray(points, dtype=np.int64)  # shape (N,3)
        N = P.shape[0]

        # Pairwise squared distances via broadcasting: (N,N,3) -> (N,N)
        D = P[:, None, :] - P[None, :, :]
        # Sum of squares along last axis
        dist2 = np.einsum("...i,...i->...", D, D)

        # Use only upper triangle (unique pairs, i<j)
        iu, ju = np.triu_indices(N, k=1)
        w = dist2[iu, ju]  # length N*(N-1)/2

        # Find indices of k smallest weights without full sort
        sel = np.argpartition(w, k - 1)[:k]  # unsorted k smallest
        sel = sel[np.argsort(w[sel])]  # optional: sort these k by weight

        # Return edges as (weight, i, j)
        return list(zip(w[sel].tolist(), iu[sel].tolist(), ju[sel].tolist()))

    def solve_part1_num(boxes, goal):
        """Solve part 1 of the puzzle with numpy."""
        edges = top_k_edges(boxes, goal)

        uf = UnionFind(len(boxes))
        # process exactly k shortest pairs
        for w, i, j in edges:
            uf.union(i, j)

        sizes = sorted(uf.get_component_sizes(), reverse=True)
        return sizes[0] * sizes[1] * sizes[2]

    answer1_num = solve_part1_num(boxes, 1000)
    print(f"Part 1: {answer1_num}")
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
                _example_boxes_p2 = parse_input(_ex.input_data)
                _result_p2 = solve_part2(_example_boxes_p2)
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
def _(UnionFind, boxes, np):
    # Solve Part 2 with numpy
    def solve_part2(boxes):
        P = np.asarray(boxes, dtype=np.int64)
        N = P.shape[0]

        D = P[:, None, :] - P[None, :, :]
        dist2 = np.einsum("...i,...i->...", D, D)

        iu, ju = np.triu_indices(N, k=1)
        w = dist2[iu, ju]

        # Sort primarily by distance, tie-break by (i, j)
        order = np.lexsort((ju, iu, w))  # last key is primary

        uf = UnionFind(N)
        components = N

        for t in order:
            i = int(iu[t])
            j = int(ju[t])
            if uf.union(i, j):
                components -= 1
                last_i, last_j = i, j
                if components == 1:
                    break

        answer2 = boxes[last_i][0] * boxes[last_j][0]

        return answer2

    answer2 = solve_part2(boxes)
    print(f"Part 2: {answer2}")
    return (solve_part2,)


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    Kruskal's algorithm:

    _build a minimum spanning tree by being relentlessly cheap and refusing to create cycles_
    """)
    return


if __name__ == "__main__":
    app.run()
