"""Four publication-quality plots saved to plots/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"truth": "#222222", "physics": "#4878CF", "mlp": "#6ACC65", "hybrid": "#D65F5F"}
ALPHA = 0.85


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(results, path="plots/trajectory.png"):
    truth = results["truth"]
    t = np.arange(len(truth)) * 0.01
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axes[i].plot(t, truth[:, i],               color=C["truth"],   lw=1.4, label="Truth")
        axes[i].plot(t, results["physics"][:, i],  color=C["physics"], lw=1.0, ls="--", label="Physics")
        axes[i].plot(t, results["mlp"][:, i],      color=C["mlp"],     lw=1.0, ls="-.", label="MLP")
        axes[i].plot(t, results["hybrid"][:, i],   color=C["hybrid"],  lw=1.0, ls=":",  label="Hybrid", alpha=ALPHA)
        axes[i].set_ylabel(label)
        if i == 0:
            axes[i].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Lorenz Trajectory — Truth vs Models", fontweight="bold")
    _save(fig, path)


def plot_phase_space(results, path="plots/phase_space.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [(0, 2, "x", "z"), (0, 1, "x", "y"), (1, 2, "y", "z")]
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        for key in ["truth", "physics", "mlp", "hybrid"]:
            r = results[key]
            ax.plot(r[:, i], r[:, j], color=C[key], lw=0.5,
                    label=key.capitalize(), alpha=ALPHA)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7)
    fig.suptitle("Phase Space Portraits", fontweight="bold")
    _save(fig, path)


def plot_prediction_vs_truth(results, path="plots/prediction_vs_truth.png"):
    mse_len = len(results["mse_physics"])
    t  = np.arange(mse_len) * 0.01
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: x-component overlay (mse_len steps, since truth has +1 for IC)
    ax = axes[0]
    truth = results["truth"]
    n = min(300, mse_len)
    ax.plot(t[:n], truth[1:n+1, 0],              color=C["truth"],   lw=1.4, label="Truth")
    ax.plot(t[:n], results["hybrid"][1:n+1, 0],  color=C["hybrid"],  lw=1.0, ls="--", label="Hybrid")
    ax.plot(t[:n], results["mlp"][1:n+1, 0],     color=C["mlp"],     lw=1.0, ls="-.", label="MLP")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("x"); ax.legend(fontsize=8)
    ax.set_title("Prediction vs Ground Truth (x-component)")

    # Right: MSE curves
    ax = axes[1]
    ax.semilogy(t, results["mse_physics"], color=C["physics"], lw=1.2, label="Physics")
    ax.semilogy(t, results["mse_mlp"],     color=C["mlp"],     lw=1.2, label="MLP")
    ax.semilogy(t, results["mse_hybrid"],  color=C["hybrid"],  lw=1.2, label="Hybrid")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("MSE (log)")
    ax.set_title("Short-Term Prediction Error"); ax.legend(fontsize=8)

    _save(fig, path)


def plot_attribution(saliency, truth, path="plots/attribution.png"):
    t   = np.arange(len(saliency)) * 0.01
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

    axes[0].plot(t, truth[:len(saliency), 0], color=C["truth"], lw=1.0)
    axes[0].set_ylabel("x(t)"); axes[0].set_title("Gradient Saliency Attribution", fontweight="bold")

    clrs = ["#e07b54", "#54a0e0", "#54c46b"]
    for d, (name, c) in enumerate(zip(["x", "y", "z"], clrs)):
        axes[1].fill_between(t, saliency[:, d], alpha=0.5, color=c, label=f"∂/∂{name}")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("|Gradient|"); axes[1].legend(fontsize=8)

    _save(fig, path)
