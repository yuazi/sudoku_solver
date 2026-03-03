"""
data.py — Sudoku dataset loader.

Downloads the sudoku dataset (features.pt + labels.pt) from powei.tw and
wraps it as a PyTorch TensorDataset.

  - Input x of shape (81, 9): One-hot encoded digits. Missing digits are
    encoded as all-zeros.
  - Target y of shape (81,): Digit indices 0-8 for the solved puzzle.

9,000 training samples and 1,000 test samples.
"""

import os
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_and_extract_archive


class Sudoku(TensorDataset):
    download_url_prefix = "https://powei.tw"
    zip_filename = "sudoku.zip"

    def __init__(self, root: str, train: bool = True):
        self.root = root
        self._folder = folder = os.path.join(root, "sudoku")
        self._fetch_data(root)

        X = torch.load(os.path.join(folder, "features.pt"), weights_only=True)
        Y = torch.load(os.path.join(folder, "labels.pt"), weights_only=True)

        # Reshape to (N, 81, 9)
        X = X.view(-1, 9 * 9, 9)
        Y = Y.view(-1, 9 * 9, 9)
        Y = Y.argmax(dim=2)  # (N, 81) — digit indices 0-8

        n_train = 9000
        if train:
            x, y = X[:n_train], Y[:n_train]
        else:
            x, y = X[n_train:], Y[n_train:]

        super().__init__(x, y)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_integrity(self) -> bool:
        for fname in ("features.pt", "labels.pt"):
            if not os.path.isfile(os.path.join(self._folder, fname)):
                return False
        return True

    def _fetch_data(self, data_dir: str) -> None:
        if self._check_integrity():
            return
        url = f"{self.download_url_prefix}/{self.zip_filename}"
        print(f"Downloading Sudoku dataset from {url} ...")
        download_and_extract_archive(
            url, data_dir, filename=self.zip_filename, remove_finished=True
        )
        print("Download complete.")
