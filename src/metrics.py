"""Evaluation: short-term MSE, divergence time, attractor similarity."""
import numpy as np
from src.lorenz import rk4_step


def rollout_nn(model, ic, n_steps):
    import torch
    model.eval()
    traj = [ic]
    s = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(n_steps):
            s = model(s)
            traj.append(s.squeeze(0).numpy())
    return np.array(traj)


def rollout_physics(ic, n_steps):
    traj = [ic]
    s = ic.copy()
    for _ in range(n_steps):
        s = rk4_step(s)
        traj.append(s)
    return np.array(traj)


def mse_curve(pred, truth):
    return np.mean((pred - truth) ** 2, axis=1)


def divergence_time(pred, truth, threshold=5.0):
    errs = np.linalg.norm(pred - truth, axis=1)
    idx  = np.where(errs > threshold)[0]
    return int(idx[0]) if len(idx) else -1


def attractor_similarity(traj_a, traj_b):
    """
    Simple attractor similarity: compare 2D histogram of (x, z) projection.
    Returns 1 - normalised L1 distance between histograms (higher = more similar).
    """
    def hist2d(t):
        H, _, _ = np.histogram2d(t[:, 0], t[:, 2], bins=30,
                                 range=[[-25, 25], [0, 55]],
                                 density=True)
        return H / H.sum()

    ha, hb = hist2d(traj_a), hist2d(traj_b)
    return float(1.0 - 0.5 * np.abs(ha - hb).sum())
