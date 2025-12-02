# Advent of Code 2025 - Python

My solutions for [Advent of Code 2025](https://adventofcode.com/2025) using modern Python data science tools.

## Tech Stack

- **[marimo](https://marimo.io/)** - Reactive Python notebooks (pure Python, git-friendly)
- **[polars](https://pola.rs/)** - Fast DataFrame library
- **[jax](https://jax.readthedocs.io/)** - High-performance numerical computing
- **[rustworkx](https://www.rustworkx.org/)** - Graph algorithms (Rust-backed)
- **[lark](https://lark-parser.readthedocs.io/)** - Parsing toolkit
- **[more-itertools](https://more-itertools.readthedocs.io/)** - Extended iteration tools
- **[advent-of-code-data](https://github.com/wimglenn/advent-of-code-data)** - Puzzle data fetching

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repo
git clone https://github.com/ballPointPenguin/aoc2025py.git
cd aoc2025py

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### AoC Session Token

To fetch your personalized puzzle inputs, you need to set your AoC session token:

1. Log into [adventofcode.com](https://adventofcode.com)
2. Open browser DevTools → Application → Cookies
3. Copy the value of the `session` cookie
4. Set it as an environment variable:

```bash
export AOC_SESSION="your_session_token_here"
```

Or create a `.env` file (already in `.gitignore`):

```
AOC_SESSION=your_session_token_here
```

## Usage

### Open a day's notebook

```bash
uv run marimo edit days/day_01.py
```

### Create a new day from template

```bash
cp days/day_01.py days/day_02.py
# Edit the DAY variable and implement your solution
```

### Run a solution as a script

```bash
uv run python days/day_01.py
```

## Project Structure

```
.
├── pyproject.toml          # Project config and dependencies
├── src/
│   └── aoc_utils/          # Shared utilities
│       ├── __init__.py
│       ├── fetching.py     # AoC data fetching
│       ├── parsing.py      # Input parsing helpers
│       └── grid.py         # 2D grid utilities
├── days/
│   ├── day_01.py           # Day 1 marimo notebook
│   ├── day_02.py           # Day 2 marimo notebook
│   └── ...
└── tests/
    └── ...
```

## Shared Utilities

### Fetching puzzle data

```python
from aoc_utils import get_puzzle, get_input, get_examples

puzzle = get_puzzle(year=2025, day=1)
raw_input = puzzle.input_data
examples = puzzle.examples
```

### Parsing helpers

```python
from aoc_utils import parse_lines, parse_ints, parse_grid, parse_sections

lines = parse_lines(raw_input)           # Split into lines
numbers = parse_ints(raw_input)          # Extract all integers
grid = parse_grid(raw_input)             # 2D character grid
sections = parse_sections(raw_input)     # Split on blank lines
```

### Grid utilities

```python
from aoc_utils import Grid, Point

grid = Grid.from_string(raw_input)
start = grid.find('S')
for neighbor, value in grid.neighbors4(start):
    print(f"{neighbor}: {value}")
```

## Tips

- marimo notebooks auto-reload when you edit `src/aoc_utils/*.py`
- Use `polars` for tabular data (much faster than pandas)
- Use `rustworkx` for graph problems (BFS, DFS, shortest paths)
- Use `lark` for complex parsing (grammars)
- Use `jax.numpy` for numerical puzzles needing speed
