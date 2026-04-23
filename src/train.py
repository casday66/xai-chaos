"""Fast training loop — converges in <10 epochs with cosine LR."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train(model, X, Y, epochs=8, lr=5e-3, batch=128):
    """Returns list of epoch losses."""
    ds     = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    history = []

    model.train()
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        history.append(total / len(ds))

    return history
