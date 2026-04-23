"""Generate, cache, and load Lorenz dataset."""
import numpy as np
from src.lorenz import simulate

DATA_PATH = "data/signals.npy"
DT = 0.01


def build_dataset(n_train=4000, n_test=500, noise_std=0.05, seed=42):
    """
    Returns dict with keys: X_train, Y_train, X_test, Y_test, test_clean.
    All arrays shape (N, 3).  Saves full noisy trajectory to DATA_PATH.
    """
    clean = simulate(n_train + n_test + 1, dt=DT, seed=seed)
    noisy = simulate(n_train + n_test + 1, dt=DT, noise_std=noise_std, seed=seed)

    np.save(DATA_PATH, noisy)

    X_train = noisy[:n_train]
    Y_train = clean[1:n_train + 1]           # predict clean next state

    X_test  = clean[n_train:n_train + n_test]
    Y_test  = clean[n_train + 1:n_train + n_test + 1]

    return {
        "X_train": X_train.astype(np.float32),
        "Y_train": Y_train.astype(np.float32),
        "X_test":  X_test.astype(np.float32),
        "Y_test":  Y_test.astype(np.float32),
        "test_clean": clean[n_train:n_train + n_test + 1],
    }
