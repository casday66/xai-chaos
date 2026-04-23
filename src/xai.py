"""Gradient saliency for the Hybrid model."""
import numpy as np
import torch
import torch.nn as nn


def gradient_saliency(model, traj):
    """
    |dMSE/d_input| for each step. Returns (T-1, 3) saliency array.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    X = torch.tensor(traj[:-1], dtype=torch.float32)
    Y = torch.tensor(traj[1:],  dtype=torch.float32)
    sal = np.empty((len(X), 3), dtype=np.float32)

    for i in range(len(X)):
        x = X[i:i+1].detach().requires_grad_(True)
        loss = loss_fn(model(x), Y[i:i+1])
        loss.backward()
        sal[i] = x.grad.abs().squeeze(0).detach().numpy()

    return sal
