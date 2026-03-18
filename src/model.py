import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import LaplacianPolynomialConv
from src.utils import NODE_FEATURE_DIM

# Odor basis — the 6 dimensions of the output vector
ODOR_BASIS = ["flowery", "fruity", "spicy", "resinous", "burnt", "putrid"]
ODOR_DIM   = len(ODOR_BASIS)


class OdorGNN(nn.Module):
    """
    Graph Neural Network that maps a molecule to a 6-dimensional odor vector.

    Architecture overview
    ─────────────────────
    1. Laplacian polynomial GNN layers
       - Each layer applies P(L) = Σ aₖLᵏ to the node feature matrix,
         then a linear transform + ReLU.
       - Degree K=2 means each node is influenced by nodes up to 2 bonds away.

    2. Adjacency tensor construction
       - For every edge (i, j) in the molecule, concatenate the final node
         embeddings: edge_feat = [hᵢ ‖ hⱼ]  →  shape [num_edges, 2·hidden]
       - This encodes bond-level chemistry (which atoms are bonded together
         and what those atoms look like after message passing).

    3. 1-D convolutions over edge features
       - Treat edges as a sequence: [1, 2·hidden, num_edges]
       - Two Conv1d layers (kernel=1, i.e. per-edge MLPs) learn edge-level
         representations independent of graph size.
       - Global mean pooling collapses the variable-length edge axis to a
         fixed-size molecule embedding.

    4. Output head
       - Two linear layers map the molecule embedding to a 6-vector.
       - No final activation — the output is an unbounded linear combination
         of the six odor basis dimensions.

    Inputs to forward():
        x : [N, 5]    node features  (element, aromatic, isotope, hcount, charge)
        L : [N, N]    graph Laplacian
        A : [N, N]    adjacency matrix

    Output:
        [6]  odor coordinates  (flowery, fruity, spicy, resinous, burnt, putrid)
    """

    def __init__(
        self,
        hidden_dim:    int = 32,
        num_gnn_layers: int = 2,
        poly_degree:   int = 2,
        conv_channels: int = 64,
        output_dim:    int = ODOR_DIM,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ── 1. Laplacian polynomial GNN layers ────────────────────────────────
        gnn_layers = []
        in_dim = NODE_FEATURE_DIM
        for _ in range(num_gnn_layers):
            gnn_layers.append(
                LaplacianPolynomialConv(in_dim, hidden_dim, K=poly_degree)
            )
            in_dim = hidden_dim
        self.gnn_layers = nn.ModuleList(gnn_layers)

        # ── 2 & 3. Conv1d over adjacency tensor (edge features) ───────────────
        # Edge feature dim = 2 * hidden_dim  (both endpoint embeddings stacked)
        edge_dim = 2 * hidden_dim

        self.edge_convs = nn.Sequential(
            nn.Conv1d(edge_dim, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
        )

        # ── 4. Output head ─────────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(conv_channels, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,   # [N, NODE_FEATURE_DIM]
        L: torch.Tensor,   # [N, N]
        A: torch.Tensor,   # [N, N]
    ) -> torch.Tensor:

        # ── Stage 1: Laplacian GNN ─────────────────────────────────────────────
        h = x
        for layer in self.gnn_layers:
            h = layer(h, L)                  # [N, hidden_dim]

        # ── Stage 2: Build adjacency tensor ───────────────────────────────────
        # Collect indices of all edges (non-zero entries of A)
        edge_idx = A.nonzero(as_tuple=False)  # [E, 2]

        if edge_idx.shape[0] == 0:
            # Edge case: isolated single atom (no bonds).
            # Treat the atom as a self-edge to keep a valid input.
            edge_feats = torch.cat([h, h], dim=1)   # [N, 2·hidden]
        else:
            i_idx = edge_idx[:, 0]
            j_idx = edge_idx[:, 1]
            # Stack the two endpoint embeddings for every edge
            edge_feats = torch.cat(
                [h[i_idx], h[j_idx]], dim=1
            )                                        # [E, 2·hidden]

        # ── Stage 3: 1-D convolutions over edge features ──────────────────────
        # Conv1d expects [batch, channels, length]
        e = edge_feats.T.unsqueeze(0)       # [1, 2·hidden, E]
        e = self.edge_convs(e)              # [1, conv_channels, E]

        # Global mean pool over the edge axis → molecule-level embedding
        pooled = e.mean(dim=2).squeeze(0)   # [conv_channels]

        # ── Stage 4: Output head → 6-vector ───────────────────────────────────
        return self.output_head(pooled)     # [ODOR_DIM]

    # ── Convenience ───────────────────────────────────────────────────────────

    def predict_named(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        A: torch.Tensor,
    ) -> dict:
        """Return a dict mapping each odor name to its predicted coordinate."""
        with torch.no_grad():
            vec = self.forward(x, L, A).cpu()
        return dict(zip(ODOR_BASIS, vec.tolist()))