# GNN Sudoku Solver

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/yuazi/sudoku_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/yuazi/sudoku_solver/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`sudoku_solver` is a PyTorch project that models Sudoku as a graph and trains a graph neural network to predict digits through message passing. It comes with a command-line solver that checks its input, an exact fallback search, Matplotlib views of each round, and tests.

## Features

- Graph construction for 9x9 Sudoku boards with 81 nodes and 1,620 directed constraint edges.
- A message-passing GNN with shared MLP messages, summed neighbor aggregation, GRU state updates, and per-cell digit logits.
- The loss is averaged over every message-passing round, so intermediate predictions are trained too.
- CLI puzzle validation before model loading, including length checks, invalid characters, contradictory givens, and clue preservation.
- An exact backtracking fallback (most constrained cell first) when the network's answer breaks a rule.
- Matplotlib visualizations for predicted solutions and step-by-step iteration output.
- Unit tests for validation, solution checking, clue preservation, the fallback solver, model stability and CLI errors.

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

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

The tests need neither the dataset nor a trained checkpoint.

## Train

```bash
python3 train.py
```

The first training run downloads the SATNet Sudoku dataset into `data/sudoku/`, trains for 30 epochs, and writes the best checkpoint to `gnn_sudoku.pth`.

Useful options:

```bash
python3 train.py --epochs 50 --batch-size 32 --lr 0.001 --n-iters 7 --hidden-dim 64 --save my_model.pth
```

Git ignores the dataset and checkpoints:

- `data/sudoku/`
- `*.pth`

## Solve a puzzle

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

## Validation

The CLI rejects invalid input before loading model weights:

- puzzle length must be exactly `81`
- characters must be `0-9` or `.` before normalization
- givens cannot contradict row, column, or 3x3 box constraints
- final reported solutions must preserve all givens

If the GNN's answer is not a valid solution, `solve.py` runs an exact backtracking search and says so in its output. If that also finds nothing, it prints the invalid board and exits with status 1.

## Project structure

```text
.
├── data.py                  # Dataset download and loading
├── model.py                 # Sudoku graph topology, batching, and GNN model
├── train.py                 # Training loop, validation split, checkpoint saving
├── solve.py                 # CLI inference, validation, fallback solving
├── visualize.py             # Matplotlib board and iteration views
├── tests/
│   ├── test_model.py
│   ├── test_solve_cli.py
│   └── test_solve_validation.py
├── requirements.txt
└── pyproject.toml
```

## Dataset

The project uses the SATNet Sudoku dataset hosted at `powei.tw` and downloads it on the first training run. The dataset contains 9,000 training puzzles and 1,000 test puzzles after the local split used by the loader.

## Results

Measured on 2026-10-25 with the default settings of `python3 train.py`: seed 42, 30 epochs, batch size 16, 7 message-passing rounds, hidden size 64 (79,753 parameters). Training took about 19 minutes on an Apple M1 Pro (MPS). Scores are on the 1,000 test puzzles, which the model never sees during training. Each test puzzle has 32 to 41 givens (36.2 on average).

| Rounds at inference     | Blank cells correct | Puzzles solved by the GNN |
| ----------------------- | ------------------- | ------------------------- |
| 1                       | 58.6%               | 0.0%                      |
| 3                       | 98.4%               | 60.0%                     |
| 7 (as trained)          | 99.95%              | 99.3%                     |
| 16 (`solve.py` default) | 99.98%              | 99.8%                     |
| 64                      | 99.98%              | 99.8%                     |

At 16 rounds the GNN solves 998 of the 1,000 test puzzles, and the backtracking fallback in `solve.py` solves the other 2. Inference takes about 4 ms per puzzle on the M1 Pro CPU with all 1,000 puzzles in one batch.

The weights are shared across rounds, so the model can run more rounds at inference than it was trained with, and extra rounds help until accuracy levels off. These numbers come from a single training run with one seed, so other hardware and seeds will differ slightly.

## License

MIT License. See [LICENSE](LICENSE) for details.
