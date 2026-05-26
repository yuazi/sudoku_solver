# GNN Sudoku Solver

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/yuazi/sudoku_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/yuazi/sudoku_solver/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`sudoku_solver` is a PyTorch project that models Sudoku as a graph and trains a graph neural network to predict digits through message passing. It includes a command-line solver, input validation, visualizations, and regression tests around the parts most likely to fail in real usage.

## What It Demonstrates

- Graph construction for 9x9 Sudoku boards with 81 nodes and 1,620 directed constraint edges.
- A message-passing GNN with shared MLP messages, summed neighbor aggregation, GRU state updates, and per-cell digit logits.
- Training over all GNN iterations so intermediate predictions are supervised, not only the final output.
- CLI puzzle validation before model loading, including length checks, invalid characters, contradictory givens, and clue preservation.
- Optional exact backtracking fallback when the neural prediction is invalid.
- Matplotlib visualizations for predicted solutions and step-by-step iteration output.
- Unit tests for validation, solution checking, clue preservation, and CLI error handling.

## Architecture

Each 9x9 board is represented as a graph:

| Concept | Implementation |
| --- | --- |
| Node | One Sudoku cell, `81` nodes total |
| Edge | Directed connection between cells sharing a row, column, or 3x3 box |
| Edge count | `1,620` directed edges per puzzle |
| Node input | Digit index, where `0` is an empty cell |
| Output | Logits for digits `1` through `9` at every cell |

The model runs `n_iters` message-passing rounds:

1. Build messages for every edge with a shared MLP over source and destination hidden states.
2. Sum incoming messages at each destination node.
3. Update node states with a `GRUCell`.
4. Emit per-node digit logits with a linear output head.

## Installation

```bash
git clone https://github.com/yuazi/sudoku_solver.git
cd sudoku_solver
python3 -m pip install -r requirements.txt
```

## Verify Locally

```bash
python3 -m unittest discover -s tests -v
```

The current test suite covers the solver validation path and can run without downloading the Sudoku dataset or training a checkpoint.

## Train

```bash
python3 train.py
```

The first training run downloads the SATNet Sudoku dataset into `data/sudoku/`, trains for 30 epochs, and writes the best checkpoint to `gnn_sudoku.pth`.

Useful options:

```bash
python3 train.py --epochs 50 --batch-size 32 --lr 0.001 --n-iters 7 --hidden-dim 64 --save my_model.pth
```

Local datasets and model checkpoints are intentionally ignored by Git:

- `data/sudoku/`
- `*.pth`

## Solve A Puzzle

Provide an 81-character string where `0` or `.` means an empty cell:

```bash
python3 solve.py "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Use a custom model checkpoint:

```bash
python3 solve.py --model my_model.pth "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Show visual output:

```bash
python3 solve.py --steps "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
python3 solve.py --plot "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

If no puzzle is provided, `solve.py` uses a built-in example.

## Validation Behavior

The CLI rejects invalid input before loading model weights:

- puzzle length must be exactly `81`
- characters must be `0-9` or `.` before normalization
- givens cannot contradict row, column, or 3x3 box constraints
- final reported solutions must preserve all givens

If the GNN prediction is not a valid Sudoku solution, the solver attempts an exact backtracking fallback and reports that path in the terminal output.

## Project Structure

```text
.
├── data.py                  # Dataset download/loading wrapper
├── model.py                 # Sudoku graph topology, batching, and GNN model
├── train.py                 # Training loop, validation split, checkpoint saving
├── solve.py                 # CLI inference, validation, fallback solving
├── visualize.py             # Matplotlib board and iteration views
├── tests/
│   ├── test_solve_cli.py
│   └── test_solve_validation.py
├── requirements.txt
└── pyproject.toml
```

## Dataset

The project uses the SATNet Sudoku dataset hosted at `powei.tw` and downloads it on the first training run. The dataset contains 9,000 training puzzles and 1,000 test puzzles after the local split used by the loader.

## Results

The training script reports validation solved fraction per epoch and final test solved fraction after loading the best checkpoint. Results can vary by hardware, seed, and training length, so the repository does not hard-code a single headline accuracy number.

## License

MIT License. See [LICENSE](LICENSE) for details.
