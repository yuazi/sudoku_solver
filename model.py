"""
model.py — Graph Neural Network for Sudoku solving.

Architecture (mirrors the reference project 12_gnn-AbdulAzizYusuf):

  Each 9x9 Sudoku is represented as a graph with 81 nodes (one per cell).
  Edges connect every pair of cells that share a row, column, or 3x3 box
  (1,620 directed edges per puzzle).

  The GNN runs n_iters message-passing / GRU-update iterations:
    1. Compute messages along every edge using msg_net (an MLP on
       concatenated source + destination hidden states).
    2. Aggregate (sum) incoming messages at each destination node.
    3. Update hidden state via a GRUCell whose input is (node_input ∥ aggregated_msg).
    4. Compute per-node digit logits via a linear output layer.

  The network returns outputs for ALL iterations so the training loss can
  be averaged over them (faster convergence).
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
#  Graph topology helper
# ---------------------------------------------------------------------------

def sudoku_edges() -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Return the directed edges of the Sudoku constraint graph.

    Each node corresponds to one cell with id = row * 9 + col (0-80).
    Two nodes are connected when they share a row, column, or 3x3 box.

    Returns:
        src_ids: LongTensor of length 1620 — source node ids.
        dst_ids: LongTensor of length 1620 — destination node ids.
    """
    src_ids: list[int] = []
    dst_ids: list[int] = []

    for node in range(81):
        row = node // 9
        col = node % 9
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        # Same row
        for c in range(9):
            neighbor = row * 9 + c
            if neighbor != node:
                src_ids.append(node)
                dst_ids.append(neighbor)

        # Same column (excluding already-added row neighbors)
        row_neighbors = set(row * 9 + c for c in range(9))
        for r in range(9):
            neighbor = r * 9 + col
            if neighbor != node and neighbor not in row_neighbors:
                src_ids.append(node)
                dst_ids.append(neighbor)

        # Same 3x3 box (excluding already-added row and column neighbors)
        col_neighbors = set(r * 9 + col for r in range(9))
        all_seen = row_neighbors | col_neighbors
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                neighbor = r * 9 + c
                if neighbor != node and neighbor not in all_seen:
                    src_ids.append(node)
                    dst_ids.append(neighbor)

    return torch.LongTensor(src_ids), torch.LongTensor(dst_ids)


# ---------------------------------------------------------------------------
#  Collate function for mini-batching graphs
# ---------------------------------------------------------------------------

# Pre-compute edges once (module-level)
_SRC_IDS, _DST_IDS = sudoku_edges()


def collate(list_of_samples):
    """Merge a list of (inputs, targets) samples into one large graph.

    Args:
        list_of_samples: list of (x, y) tuples where
            x of shape (81, 9) — one-hot encoded puzzle cells.
            y of shape (81,)  — target digit indices (0-8).

    Returns:
        inputs  of shape (B*81, 9)
        targets of shape (B*81,)
        src_ids of shape (B*1620,)
        dst_ids of shape (B*1620,)
    """
    batch_size = len(list_of_samples)

    inputs = torch.cat([s[0] for s in list_of_samples], dim=0)
    targets = torch.cat([s[1] for s in list_of_samples], dim=0)

    src_list, dst_list = [], []
    for i in range(batch_size):
        offset = i * 81
        src_list.append(_SRC_IDS + offset)
        dst_list.append(_DST_IDS + offset)

    src_ids = torch.cat(src_list, dim=0)
    dst_ids = torch.cat(dst_list, dim=0)

    return inputs, targets, src_ids, dst_ids


# ---------------------------------------------------------------------------
#  GNN model
# ---------------------------------------------------------------------------

class GNN(nn.Module):
    """Graph Neural Network Sudoku solver.

    Args:
        n_iters:          Number of message-passing iterations (default 7).
        n_node_features:  Dimension of each node's hidden state (default 10).
        n_node_inputs:    Input feature size per node — 9 (one-hot digit or zeros).
        n_edge_features:  Message vector size (default 11).
        n_node_outputs:   Output logits per node — 9 (one per digit).
    """

    def __init__(
        self,
        n_iters: int = 7,
        n_node_features: int = 10,
        n_node_inputs: int = 9,
        n_edge_features: int = 11,
        n_node_outputs: int = 9,
    ):
        super().__init__()

        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_edge_features = n_edge_features
        self.n_node_outputs = n_node_outputs

        # Message network: MLP(src_state ∥ dst_state) → message vector
        self.msg_net = nn.Sequential(
            nn.Linear(2 * n_node_features, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, n_edge_features),
        )

        # GRU cell: updates hidden state from (node_input ∥ aggregated_messages)
        self.gru = nn.GRUCell(n_node_inputs + n_edge_features, n_node_features)

        # Output head: project hidden state to digit logits
        self.output_layer = nn.Linear(n_node_features, n_node_outputs)

    # ------------------------------------------------------------------

    def forward(
        self,
        node_inputs: torch.Tensor,
        src_ids: torch.LongTensor,
        dst_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Run message-passing for n_iters iterations.

        Args:
            node_inputs: (n_nodes, n_node_inputs) — one-hot puzzle input.
            src_ids:     (n_edges,) — source node indices.
            dst_ids:     (n_edges,) — destination node indices.

        Returns:
            outputs: (n_iters, n_nodes, n_node_outputs) — logits at each iter.
        """
        n_nodes = node_inputs.size(0)
        device = node_inputs.device

        # Hidden states initialised to zero
        h = torch.zeros(n_nodes, self.n_node_features, device=device)

        outputs = []
        for _ in range(self.n_iters):
            # 1. Compute messages
            msg_input = torch.cat([h[src_ids], h[dst_ids]], dim=1)  # (E, 2*H)
            messages = self.msg_net(msg_input)                       # (E, F)

            # 2. Aggregate (sum) messages at destination nodes
            agg = torch.zeros(n_nodes, self.n_edge_features, device=device)
            agg.index_add_(0, dst_ids, messages)                     # (N, F)

            # 3. Update hidden state via GRU
            gru_input = torch.cat([node_inputs, agg], dim=1)         # (N, I+F)
            h = self.gru(gru_input, h)                               # (N, H)

            # 4. Produce output logits
            outputs.append(self.output_layer(h))                     # (N, 9)

        return torch.stack(outputs, dim=0)  # (n_iters, N, 9)
