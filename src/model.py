import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import LaplacianPolynomialConv
from src.utils import NODE_FEATURE_DIM


class OdorGNN(nn.Module):
    def __init__(
        self,
        hidden_dim:    int = 32,
        num_gnn_layers: int = 2,
        poly_degree:   int = 2,
        conv_channels: int = 64,
        output_dim:    int = 138,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        gnn_layers = []
        in_dim = NODE_FEATURE_DIM
        for _ in range(num_gnn_layers):
            gnn_layers.append(
                LaplacianPolynomialConv(in_dim, hidden_dim, K=poly_degree)
            )
            in_dim = hidden_dim
        self.gnn_layers = nn.ModuleList(gnn_layers)

        edge_dim = 2 * hidden_dim

        self.edge_convs = nn.Sequential(
            nn.Conv1d(edge_dim, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
        )

        self.output_head = nn.Sequential(
            nn.Linear(conv_channels, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )


    def forward(
        self,
        x: torch.Tensor,   # [N, NODE_FEATURE_DIM]
        L: torch.Tensor,   # [N, N]
        A: torch.Tensor,   # [N, N]
    ) -> torch.Tensor:

        h = x
        for layer in self.gnn_layers:
            h = layer(h, L)                  # [N, hidden_dim]

        edge_idx = A.nonzero(as_tuple=False)  # [E, 2]

        if edge_idx.shape[0] == 0:

            edge_feats = torch.cat([h, h], dim=1)
        else:
            i_idx = edge_idx[:, 0]
            j_idx = edge_idx[:, 1]
            edge_feats = torch.cat(
                [h[i_idx], h[j_idx]], dim=1
            ) 

        e = edge_feats.T.unsqueeze(0)
        e = self.edge_convs(e)

        pooled = e.mean(dim=2).squeeze(0)

        return self.output_head(pooled)