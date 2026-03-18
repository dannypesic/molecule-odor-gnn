import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPolynomialConv(nn.Module):
    """
    A single graph convolution layer using a learnable K-degree Laplacian polynomial.

    The polynomial is:
        P(L) = sum_{k=0}^{K}  a_k * L^k

    where a_0 … a_K are learnable scalar weights and L^0 = I (identity).
    k=0 keeps the node's own features, k=1 pulls in immediate neighbours,
    k=2 reaches two hops away — appropriate for local molecular chemistry.

    The full layer update is:
        H_out = activation( P(L) @ H_in @ W + b )

    where W is a [in_features, out_features] weight matrix and b a bias vector.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 2):
        """
        Args:
            in_features:  dimensionality of input node features
            out_features: dimensionality of output node features
            K:            polynomial degree (2 recommended for molecules)
        """
        super().__init__()
        self.K = K

        # Learnable polynomial coefficients: one scalar per degree 0..K
        # Initialised to 1/(K+1) so P(L) ≈ average of L^k at the start
        self.poly_weights = nn.Parameter(
            torch.full((K + 1,), 1.0 / (K + 1))
        )

        # Linear projection applied after the polynomial convolution
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : node feature matrix  [N, in_features]
            L : Laplacian matrix     [N, N]

        Returns:
            Tensor [N, out_features]
        """
        N = L.shape[0]
        device = L.device

        # Build P(L) = sum_k  a_k * L^k  iteratively to avoid storing all powers
        L_power = torch.eye(N, device=device)   # L^0 = I
        P = self.poly_weights[0] * L_power

        for k in range(1, self.K + 1):
            L_power = L_power @ L               # L^k
            P = P + self.poly_weights[k] * L_power

        # Graph convolution: spread features through the polynomial
        out = P @ x                # [N, in_features]

        # Linear transform to target dimension
        out = self.linear(out)     # [N, out_features]

        return F.relu(out)