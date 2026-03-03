"""
solve.py — CLI to solve a Sudoku puzzle with the trained GNN.

Usage:
    # Solve a puzzle given as an 81-char string ('0' or '.' = empty)
    python solve.py "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

    # Show step-by-step GNN iteration process
    python solve.py --steps "530070000..."

    # Use a custom model file
    python solve.py --model my_model.pth "530070000..."
"""

import argparse
import sys
from typing import Optional
import torch
from model import GNN, sudoku_edges
from visualize import draw_sudoku, draw_solution_steps, board_from_string


# ---------------------------------------------------------------------------
#  Solver logic
# ---------------------------------------------------------------------------

def load_model(model_path: str, n_iters: int = 7) -> GNN:
    gnn = GNN(n_iters=n_iters)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    gnn.load_state_dict(state)
    gnn.eval()
    return gnn


def solve_puzzle(gnn, puzzle_str: str):
    """Run the GNN on a puzzle string and return (outputs, solution_str).

    Args:
        gnn:        Trained GNN model.
        puzzle_str: 81-char string ('0'/'.' = empty, '1'-'9' = known digit).

    Returns:
        outputs:      (n_iters, 81, 9) tensor - logits at each iteration.
        solution_str: 81-char string of the predicted solution.
        is_valid:     Whether the solution satisfies sudoku constraints and preserves givens.
    """
    x = board_from_string(puzzle_str)          # (81, 9)
    src_ids, dst_ids = sudoku_edges()

    with torch.no_grad():
        outputs = gnn(x, src_ids, dst_ids)     # (n_iters, 81, 9)

    pred = outputs[-1].argmax(dim=1) + 1       # digits 1-9, shape (81,)

    # Enforce fixed clues from the input puzzle so givens are never overwritten.
    given_mask = x.sum(dim=1) > 0
    if given_mask.any():
        given_digits = x[given_mask].argmax(dim=1) + 1
        pred[given_mask] = given_digits

    solution_str = "".join(str(d.item()) for d in pred)
    is_valid = _check_solution(solution_str) and _respects_givens(puzzle_str, solution_str)
    return outputs, solution_str, is_valid


def _validate_puzzle_input(puzzle_str: str) -> Optional[str]:
    """Return an error string when input is invalid, otherwise None."""
    if len(puzzle_str) != 81:
        return f"puzzle must be 81 characters, got {len(puzzle_str)}"
    if any(ch not in "0123456789" for ch in puzzle_str):
        return "puzzle must contain only digits 0-9 (or '.' before normalization)"
    if not _givens_are_consistent(puzzle_str):
        return "puzzle givens are contradictory"
    return None


def _givens_are_consistent(puzzle_str: str) -> bool:
    """Check that known clues do not violate row/column/box constraints."""
    grid = [puzzle_str[i * 9:(i + 1) * 9] for i in range(9)]

    def has_duplicate(group: list[str]) -> bool:
        vals = [v for v in group if v != "0"]
        return len(vals) != len(set(vals))

    for row in grid:
        if has_duplicate(list(row)):
            return False

    for c in range(9):
        col = [grid[r][c] for r in range(9)]
        if has_duplicate(col):
            return False

    for br in range(3):
        for bc in range(3):
            box = [grid[br * 3 + r][bc * 3 + c] for r in range(3) for c in range(3)]
            if has_duplicate(box):
                return False

    return True


def _respects_givens(puzzle_str: str, sol: str) -> bool:
    """Ensure fixed clues in puzzle are unchanged in the predicted solution."""
    puzzle_str = puzzle_str.replace(".", "0").strip()
    for p, s in zip(puzzle_str, sol):
        if p != "0" and p != s:
            return False
    return True


def _check_solution(sol: str) -> bool:
    """Basic validity check: all rows, columns, and boxes contain 1-9."""
    digits = [int(c) for c in sol]
    grid = [digits[i*9:(i+1)*9] for i in range(9)]

    def valid_group(group):
        return sorted(group) == list(range(1, 10))

    # Rows
    for row in grid:
        if not valid_group(row):
            return False
    # Columns
    for c in range(9):
        if not valid_group([grid[r][c] for r in range(9)]):
            return False
    # 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = [grid[br*3+r][bc*3+c] for r in range(3) for c in range(3)]
            if not valid_group(box):
                return False
    return True


def _pretty_print(puzzle_str: str, solution_str: str) -> None:
    """Print both boards side-by-side in the terminal."""
    def fmt(s, mark_zeros=True):
        lines = []
        for i, ch in enumerate(s):
            if i % 27 == 0 and i > 0:
                lines.append("├───────┼───────┼───────┤")
            if i % 9 == 0:
                lines.append("│")
            if i % 3 == 0 and i % 9 != 0:
                lines[-1] += " │"
            cell = "." if (mark_zeros and ch == "0") else ch
            lines[-1] += f" {cell}"
            if (i + 1) % 9 == 0:
                lines[-1] += " │"
        return lines

    p_lines = ["┌───────┬───────┬───────┐"] + fmt(puzzle_str) + ["└───────┴───────┴───────┘"]
    s_lines = ["┌───────┬───────┬───────┐"] + fmt(solution_str, mark_zeros=False) + ["└───────┴───────┴───────┘"]

    print(f"\n{'  PUZZLE':^25}      {'  SOLUTION':^25}")
    for pl, sl in zip(p_lines, s_lines):
        print(f"  {pl:<25}    {sl:<25}")


def _solve_with_backtracking(puzzle_str: str) -> Optional[str]:
    """Return an exact Sudoku solution for puzzle_str, or None if unsolved."""
    grid = [int(ch) for ch in puzzle_str.replace(".", "0").strip()]

    def allowed(idx: int, val: int) -> bool:
        r, c = divmod(idx, 9)
        row = r * 9
        if any(grid[row + cc] == val for cc in range(9)):
            return False
        if any(grid[rr * 9 + c] == val for rr in range(9)):
            return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if grid[rr * 9 + cc] == val:
                    return False
        return True

    def search() -> bool:
        try:
            idx = grid.index(0)
        except ValueError:
            return True

        for val in range(1, 10):
            if allowed(idx, val):
                grid[idx] = val
                if search():
                    return True
                grid[idx] = 0
        return False

    if search():
        return "".join(str(v) for v in grid)
    return None


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a Sudoku with the GNN")
    parser.add_argument("puzzle", nargs="?",
                        help="81-char puzzle string ('0' or '.' = empty cell)")
    parser.add_argument("--model",   default="gnn_sudoku.pth",
                        help="Path to trained model weights")
    parser.add_argument("--n-iters", type=int, default=7,
                        help="Number of GNN iterations (must match training)")
    parser.add_argument("--steps",  action="store_true",
                        help="Show iterative solution refinement (matplotlib)")
    parser.add_argument("--plot",   action="store_true",
                        help="Show puzzle + solution side by side (matplotlib)")
    args = parser.parse_args()

    # Default example puzzle if none provided
    EXAMPLE = (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    )
    puzzle_str = args.puzzle or EXAMPLE
    puzzle_str = puzzle_str.replace(".", "0").strip()

    err = _validate_puzzle_input(puzzle_str)
    if err is not None:
        print(f"ERROR: {err}")
        sys.exit(1)

    # Load model
    try:
        gnn = load_model(args.model, n_iters=args.n_iters)
    except FileNotFoundError:
        print(f"ERROR: model file '{args.model}' not found.")
        print("Run  python train.py  first to train the model.")
        sys.exit(1)

    # Solve
    outputs, solution_str, is_valid = solve_puzzle(gnn, puzzle_str)

    # Fallback to exact search if the GNN prediction is invalid.
    used_fallback = False
    if not is_valid:
        exact = _solve_with_backtracking(puzzle_str)
        if exact is not None:
            solution_str = exact
            is_valid = True
            used_fallback = True

    _pretty_print(puzzle_str, solution_str)
    if is_valid and used_fallback:
        status = "✓ VALID (exact fallback)"
    elif is_valid:
        status = "✓ VALID"
    else:
        status = "✗ INVALID (puzzle may be too hard or model needs more training)"
    print(f"\nSolution status: {status}\n")

    if args.plot or args.steps:
        import matplotlib
        matplotlib.use("TkAgg") if sys.platform != "darwin" else None
        if args.steps:
            draw_solution_steps(outputs, board_from_string(puzzle_str))
        else:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            draw_sudoku(board_from_string(puzzle_str).view(9, 9, 9),
                        logits=False, title="Puzzle", ax=ax1)
            draw_sudoku(outputs[-1].view(9, 9, 9),
                        logits=True, title="GNN Solution", ax=ax2)
            plt.suptitle("GNN Sudoku Solver", fontsize=14)
            plt.tight_layout()
            plt.show()
