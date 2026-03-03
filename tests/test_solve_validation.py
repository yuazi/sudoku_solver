import unittest

import torch

from solve import (
    _check_solution,
    _givens_are_consistent,
    _respects_givens,
    _validate_puzzle_input,
    solve_puzzle,
)


SOLVED = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


class DummyGNN:
    def __init__(self, solution: str, n_iters: int = 2):
        self.n_iters = n_iters
        logits = torch.full((81, 9), -5.0)
        for i, ch in enumerate(solution):
            logits[i, int(ch) - 1] = 5.0
        self._outputs = torch.stack([logits for _ in range(n_iters)], dim=0)

    def __call__(self, *_args, **_kwargs):
        return self._outputs


class SolveValidationTests(unittest.TestCase):
    def test_check_solution_accepts_valid_board(self):
        self.assertTrue(_check_solution(SOLVED))

    def test_check_solution_rejects_invalid_board(self):
        bad = "1" * 81
        self.assertFalse(_check_solution(bad))

    def test_givens_consistency_detects_contradiction(self):
        contradictory = "55" + "0" * 79
        self.assertFalse(_givens_are_consistent(contradictory))

    def test_validate_puzzle_rejects_non_digit_characters(self):
        invalid = "a" + "0" * 80
        err = _validate_puzzle_input(invalid)
        self.assertIsNotNone(err)
        self.assertIn("digits 0-9", err)

    def test_respects_givens_rejects_clue_mismatch(self):
        puzzle = "934678912" + "0" * 72
        self.assertFalse(_respects_givens(puzzle, SOLVED))

    def test_solve_puzzle_marks_invalid_when_givens_not_preserved(self):
        puzzle = "934678912" + "0" * 72
        gnn = DummyGNN(SOLVED)
        _outputs, _solution, is_valid = solve_puzzle(gnn, puzzle)
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()

