# GNN Sudoku Solver

A Graph Neural Network that learns to solve Sudoku puzzles, inspired by the `12_gnn-AbdulAzizYusuf` coursework.

## Architecture

Each 9×9 Sudoku board is modelled as a **graph with 81 nodes** (one per cell).  
Two cells are connected by a directed edge if they share a **row**, **column**, or **3×3 box** (1,620 edges total).

The GNN runs **7 message-passing iterations**:
1. Every node sends a message to its neighbours via a shared 3-layer MLP (`msg_net`)
2. Incoming messages are summed per node
3. A **GRUCell** updates each node's hidden state from `(cell_input ‖ aggregated_messages)`
4. A linear output layer converts hidden states to **digit logits (1–9)**

Loss = mean cross-entropy over **all** iterations → encourages fast convergence.

## Project structure

```
sudoku_solver/
├── data.py       # Dataset loader (downloads 10k puzzles from powei.tw)
├── model.py      # GNN class, edge generator, collate function
├── train.py      # Training loop (Adam, 30 epochs)
├── solve.py      # CLI inference tool
├── visualize.py  # Board rendering + iteration visualisation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Validate locally

```bash
python -m unittest discover -s tests -v
```

## Train

```bash
python train.py
# Downloads the dataset on first run (~20 MB), then trains for 30 epochs.
# Best model is saved to gnn_sudoku.pth
# Note: final solved fraction can vary by hardware/seed and training length.
```

Additional options:
```bash
python train.py --epochs 50 --batch-size 32 --lr 0.001 --n-iters 7 --save my_model.pth
```

## Solve a puzzle

```bash
# Provide an 81-char string ('0' or '.' = unknown cell, '1'-'9' = given digit)
python solve.py "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

# Show step-by-step GNN iteration with matplotlib
python solve.py --steps "530070000..."

# Show side-by-side puzzle / solution plot
python solve.py --plot "530070000..."
```

If no puzzle is provided, a built-in example is used.

`solve.py` now validates inputs before inference:
- puzzle must be exactly 81 cells
- only `0-9` (or `.` before normalization) are allowed
- contradictory givens are rejected
- a predicted board is only marked valid if it both solves Sudoku and preserves givens

## Dataset

[SATNet Sudoku dataset](https://github.com/locuslab/SATNet) hosted at `powei.tw`.  
Downloaded automatically on first run.

## Results

| Metric | Value |
|---|---|
| Test puzzles | 1,000 |
| Fraction solved (30 epochs) | run locally and report your own checkpoint metric |
| Model size | ~57 kB |

## License

This project is licensed under the MIT License - see `LICENSE`.
