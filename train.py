"""
train.py — Training loop for the GNN Sudoku solver.

Usage:
    python train.py [--data-dir ./data] [--epochs 30] [--batch-size 16]
                    [--lr 0.001] [--n-iters 7] [--save gnn_sudoku.pth]

Reproduces the setup from 12_gnn-AbdulAzizYusuf:
  - Adam optimiser, lr=0.001
  - Loss = mean cross-entropy over ALL GNN iterations (encourages fast convergence)
  - 30 epochs, batch size 16
  - Reports per-epoch train loss and fraction of solved test puzzles
"""

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Sudoku
from model import GNN, collate


# ---------------------------------------------------------------------------
#  Evaluation helper
# ---------------------------------------------------------------------------

def fraction_solved(gnn: GNN, loader: DataLoader, device: torch.device) -> float:
    """Fraction of test puzzles where the GNN's final output is entirely correct."""
    gnn.eval()
    n_total = n_solved = 0
    with torch.no_grad():
        for inputs, targets, src_ids, dst_ids in loader:
            batch_size = inputs.size(0) // 81
            inputs  = inputs.to(device)
            targets = targets.to(device)
            src_ids = src_ids.to(device)
            dst_ids = dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)         # (n_iters, B*81, 9)
            final   = outputs[-1].argmax(dim=1)             # (B*81,)
            correct = (final == targets).view(batch_size, 81).all(dim=1)  # (B,)
            n_total  += batch_size
            n_solved += correct.sum().item()

    return n_solved / n_total if n_total > 0 else 0.0


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    trainset = Sudoku(args.data_dir, train=True)
    testset  = Sudoku(args.data_dir, train=False)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             collate_fn=collate, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=args.batch_size,
                             collate_fn=collate, shuffle=False)
    print(f"Train: {len(trainset)} puzzles | Test: {len(testset)} puzzles")

    # Model
    gnn = GNN(n_iters=args.n_iters).to(device)
    print(gnn)

    optimizer = optim.Adam(gnn.parameters(), lr=args.lr)

    best_frac = 0.0
    for epoch in range(1, args.epochs + 1):
        gnn.train()
        total_loss = 0.0

        for inputs, targets, src_ids, dst_ids in trainloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            src_ids = src_ids.to(device)
            dst_ids = dst_ids.to(device)

            optimizer.zero_grad()
            outputs = gnn(inputs, src_ids, dst_ids)   # (n_iters, B*81, 9)

            # Average cross-entropy across ALL iterations
            loss = sum(F.cross_entropy(outputs[i], targets)
                       for i in range(gnn.n_iters)) / gnn.n_iters
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        frac = fraction_solved(gnn, testloader, device)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss {avg_loss:.4f} | "
              f"Solved {frac:.4f} ({frac*100:.1f}%)")

        if frac > best_frac:
            best_frac = frac
            torch.save(gnn.state_dict(), args.save)
            print(f"  → Saved best model to {args.save}  (frac={best_frac:.4f})")

    print(f"\nTraining complete. Best fraction solved: {best_frac:.4f}")
    print(f"Model saved to: {args.save}")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN Sudoku solver")
    parser.add_argument("--data-dir",   default="./data",          help="Dataset directory")
    parser.add_argument("--epochs",     type=int,   default=30,    help="Number of epochs")
    parser.add_argument("--batch-size", type=int,   default=16,    help="Mini-batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--n-iters",    type=int,   default=7,     help="GNN iterations")
    parser.add_argument("--save",       default="gnn_sudoku.pth",  help="Model save path")
    args = parser.parse_args()
    train(args)
