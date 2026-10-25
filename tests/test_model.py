import unittest

import torch

from model import GNN, sudoku_edges


class TestGNNStability(unittest.TestCase):
    def test_logits_stay_bounded_over_many_rounds(self):
        # Regression: a residual on top of the GRU (h = h + gru(...)) grew the
        # hidden state every round, so extra rounds at inference blew up the logits.
        torch.manual_seed(0)
        src_ids, dst_ids = sudoku_edges()
        with torch.no_grad():
            logits = GNN(n_iters=64)(torch.randint(0, 10, (81,)), src_ids, dst_ids)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertLess(logits.abs().max().item(), 100)


if __name__ == "__main__":
    unittest.main()
