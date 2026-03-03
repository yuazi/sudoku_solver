"""
visualize.py — Sudoku board rendering utilities.

Provides two functions:
  draw_sudoku(board, logits=False)
      Draw a sudoku board from either:
        - a (9, 9, 9) one-hot tensor  (logits=False)
        - a (9, 9, 9) logit tensor    (logits=True)  — shows probability bars

  draw_solution_steps(outputs)
      Show the GNN's output over multiple iterations side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Internal helper
# ---------------------------------------------------------------------------

def _show_proba(proba: torch.Tensor, r: int, c: int, ax) -> None:
    """Render probability bars or the argmax digit in one cell."""
    cm = plt.cm.Reds
    ix = proba.argmax()
    if proba[ix] > 0.9:
        ax.text(c + 0.5, r + 0.5, str(ix.item() + 1),
                ha="center", va="center", fontsize=22, fontweight="bold")
    else:
        for d in range(9):
            dx = dy = 1 / 6
            px = c + dx + (d // 3) * (2 * dx)
            py = r + dy + (d % 3) * (2 * dy)
            p = proba[d].item()
            ax.fill(
                [px - dx, px + dx, px + dx, px - dx, px - dx],
                [py - dy, py - dy, py + dy, py + dy, py - dy],
                color=cm(int(p * 255)),
            )
            ax.text(px, py, str(d + 1), ha="center", va="center", fontsize=6, color="white")


def _configure_axes(ax) -> None:
    ax.set(xlim=(0, 9), ylim=(9, 0),
           xticks=np.arange(10), xticklabels=[],
           yticks=np.arange(10), yticklabels=[])
    # Minor grid — cell borders
    ax.grid(True, which="minor", linewidth=0.5, color="gray")
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which="minor", length=0)
    # Major grid — box borders (thicker)
    ax.grid(True, which="major", linewidth=2.5, color="black")
    ax.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax.yaxis.set_major_locator(plt.MultipleLocator(3))
    ax.tick_params(which="major", length=0)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def draw_sudoku(board: torch.Tensor, logits: bool = False,
                title: str = "", ax=None) -> None:
    """Draw a sudoku board.

    Args:
        board:  Tensor of shape (9, 9, 9).
                  logits=False — one-hot or all-zeros for unknown cells.
                  logits=True  — raw logits; probabilities shown as colour bars.
        logits: Whether `board` contains logits (True) or one-hot (False).
        title:  Optional title for the axes.
        ax:     Existing matplotlib Axes to draw on; creates a new figure if None.
    """
    assert board.shape == (9, 9, 9), f"Expected shape (9,9,9), got {board.shape}"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, figsize=(7, 7))

    _configure_axes(ax)
    if title:
        ax.set_title(title, fontsize=14, pad=8)

    with torch.no_grad():
        if logits:
            probs = F.softmax(board, dim=2)
            for r in range(9):
                for c in range(9):
                    _show_proba(probs[r, c], r, c, ax)
        else:
            for r in range(9):
                for c in range(9):
                    ix = board[r, c].nonzero(as_tuple=False)
                    if ix.numel() > 0:
                        digit = ix.item() + 1  # digits 1-9 for display
                        ax.text(c + 0.5, r + 0.5, str(digit),
                                ha="center", va="center", fontsize=22,
                                fontweight="bold")

    if standalone:
        plt.tight_layout()
        plt.show()


def draw_solution_steps(outputs: torch.Tensor, puzzle: torch.Tensor,
                        max_steps: int = 7) -> None:
    """Visualise the GNN's progressive solution across iterations.

    Args:
        outputs: (n_iters, 81, 9) — network logits at each iteration.
        puzzle:  (81, 9)          — original (partial) puzzle.
        max_steps: How many iterations to display.
    """
    n_iters = min(outputs.shape[0], max_steps)
    ncols = n_iters + 1  # +1 for the original puzzle

    fig = plt.figure(figsize=(ncols * 3, 3.5))
    gs = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.05)

    # --- Original puzzle
    ax0 = fig.add_subplot(gs[0, 0])
    draw_sudoku(puzzle.view(9, 9, 9), logits=False, title="Puzzle", ax=ax0)

    # --- GNN iterations
    for i in range(n_iters):
        ax = fig.add_subplot(gs[0, i + 1])
        board = outputs[i].view(9, 9, 9)
        draw_sudoku(board, logits=True, title=f"Iter {i + 1}", ax=ax)

    plt.suptitle("GNN Sudoku Solver — Iterative Refinement", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def board_from_string(puzzle_str: str) -> torch.Tensor:
    """Convert a flat 81-character string to a (81, 9) one-hot tensor.

    Use '0' or '.' to denote empty cells, '1'-'9' for known digits.

    Example:
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    """
    puzzle_str = puzzle_str.replace(".", "0").strip()
    assert len(puzzle_str) == 81, f"Expected 81 chars, got {len(puzzle_str)}"
    x = torch.zeros(81, 9)
    for i, ch in enumerate(puzzle_str):
        d = int(ch)
        if d > 0:
            x[i, d - 1] = 1.0
    return x
