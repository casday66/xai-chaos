"""xai-chaos: Hybrid Physics-AI for Chaotic Systems — main entry point."""
import os
import numpy as np
import pandas as pd
import torch

torch.manual_seed(42)
np.random.seed(42)

from src.dataset  import build_dataset
from src.models   import PhysicsBaseline, MLP, HybridModel
from src.train    import train
from src.metrics  import rollout_nn, rollout_physics, mse_curve, divergence_time, attractor_similarity
from src.xai      import gradient_saliency
from src.viz      import plot_trajectory, plot_phase_space, plot_prediction_vs_truth, plot_attribution

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

N_TEST = 500

# ── 1. Data ──────────────────────────────────────────────────────────────────
print("[1/6] Generating dataset …")
ds = build_dataset(n_train=4000, n_test=N_TEST, noise_std=0.05)
print(f"      Saved → data/signals.npy  ({np.load('data/signals.npy').shape})")

# ── 2. Train ─────────────────────────────────────────────────────────────────
print("[2/6] Training models (8 epochs each) …")
mlp    = MLP(hidden=32)
hybrid = HybridModel(hidden=32)
mlp_loss    = train(mlp,    ds["X_train"], ds["Y_train"])
hybrid_loss = train(hybrid, ds["X_train"], ds["Y_train"])
print(f"      MLP final loss:    {mlp_loss[-1]:.6f}")
print(f"      Hybrid final loss: {hybrid_loss[-1]:.6f}")

# ── 3. Rollout predictions ────────────────────────────────────────────────────
print("[3/6] Evaluating …")
ic    = ds["test_clean"][0]
truth = ds["test_clean"]

phys_roll   = rollout_physics(ic, N_TEST)
mlp_roll    = rollout_nn(mlp,    ic, N_TEST)
hybrid_roll = rollout_nn(hybrid, ic, N_TEST)

results = {
    "truth":        truth,
    "physics":      phys_roll,
    "mlp":          mlp_roll,
    "hybrid":       hybrid_roll,
    "mse_physics":  mse_curve(phys_roll[1:],   truth[1:]),
    "mse_mlp":      mse_curve(mlp_roll[1:],    truth[1:]),
    "mse_hybrid":   mse_curve(hybrid_roll[1:], truth[1:]),
}

rows = []
for name, pred in [("physics", phys_roll), ("mlp", mlp_roll), ("hybrid", hybrid_roll)]:
    rows.append({
        "model":              name,
        "mean_mse":           float(results[f"mse_{name}"].mean()),
        "divergence_step":    divergence_time(pred, truth),
        "attractor_sim":      attractor_similarity(pred, truth),
        "final_train_loss":   mlp_loss[-1] if name == "mlp" else (hybrid_loss[-1] if name == "hybrid" else float("nan")),
    })
    print(f"      {name:8s}  MSE={rows[-1]['mean_mse']:.4f}  "
          f"div@step={rows[-1]['divergence_step']:>4d}  "
          f"attractor_sim={rows[-1]['attractor_sim']:.3f}")

pd.DataFrame(rows).to_csv("outputs/results.csv", index=False)
print("      Saved → outputs/results.csv")

# ── 4. XAI ───────────────────────────────────────────────────────────────────
print("[4/6] Computing gradient saliency …")
saliency = gradient_saliency(hybrid, truth)
attr_mean = saliency.mean(axis=0)
print(f"      Mean attribution — x:{attr_mean[0]:.4f}  y:{attr_mean[1]:.4f}  z:{attr_mean[2]:.4f}")
pd.DataFrame(saliency, columns=["sal_x", "sal_y", "sal_z"]).to_csv("outputs/saliency.csv", index=False)

# ── 5. Plots ─────────────────────────────────────────────────────────────────
print("[5/6] Generating plots …")
plot_trajectory(results)
plot_phase_space(results)
plot_prediction_vs_truth(results)
plot_attribution(saliency, truth)
for f in os.listdir("plots"):
    kb = os.path.getsize(f"plots/{f}") // 1024
    print(f"      plots/{f}  {kb} KB")

# ── 6. Training loss CSV ─────────────────────────────────────────────────────
print("[6/6] Saving training history …")
pd.DataFrame({
    "epoch": range(1, len(mlp_loss)+1),
    "mlp_loss": mlp_loss,
    "hybrid_loss": hybrid_loss,
}).to_csv("outputs/training_loss.csv", index=False)

print("\n✓ All done — see outputs/ and plots/")
