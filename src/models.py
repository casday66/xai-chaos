"""Three models: PhysicsBaseline, MLP, HybridModel."""
import numpy as np
import torch
import torch.nn as nn
from src.lorenz import rk4_step

DT = 0.01
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


class PhysicsBaseline:
    def predict_batch(self, X):          # X: np (N,3) → np (N,3)
        return np.array([rk4_step(x, DT) for x in X])


def _mlp_block(hidden=32):
    return nn.Sequential(
        nn.Linear(3, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 3),
    )


class MLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = _mlp_block(hidden)

    def forward(self, x):
        return self.net(x)


class HybridModel(nn.Module):
    """Differentiable RK4 physics step + learned residual."""

    def __init__(self, hidden=32):
        super().__init__()
        self.residual = _mlp_block(hidden)

    def _physics(self, x):
        # Differentiable RK4 (4th-order)
        def deriv(s):
            sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
            return torch.stack([
                SIGMA * (sy - sx),
                sx * (RHO - sz) - sy,
                sx * sy - BETA * sz,
            ], dim=1)

        k1 = deriv(x)
        k2 = deriv(x + .5 * DT * k1)
        k3 = deriv(x + .5 * DT * k2)
        k4 = deriv(x + DT * k3)
        return x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def forward(self, x):
        return self._physics(x) + self.residual(x)
