import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPolynomialConv(nn.Module):
    # P(L) = sum_{k=0}^{K}  a_k * L^k
    # H_out = activation( P(L) @ H_in @ W + b )
    # where W is a [in_features, out_features] weight matrix and b a bias vector.

    def __init__(self, in_features: int, out_features: int, K: int = 2):
        # in_features:  dimensionality of input node features
        # out_features: dimensionality of output node features
        # K:            polynomial degree (2 recommended for molecules)

        super().__init__()
        self.K = K

        self.poly_weights = nn.Parameter(
            torch.full((K + 1,), 1.0 / (K + 1))
        )

        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:

        N = L.shape[0]
        device = L.device

        L_power = torch.eye(N, device=device)
        P = self.poly_weights[0] * L_power

        for k in range(1, self.K + 1):
            L_power = L_power @ L
            P = P + self.poly_weights[k] * L_power

        out = P @ x

        out = self.linear(out)

        return F.relu(out)