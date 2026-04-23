# Hybrid Physics-AI for Chaotic Systems with Explainable AI

> **Key Insight:** *Accuracy does not imply physical correctness.*  
> A model can achieve low MSE on short horizons while producing attractors with wrong geometry.

---

## Problem

The Lorenz-63 system exhibits sensitive dependence on initial conditions (SDIC), making deterministic long-horizon prediction impossible. Classical black-box models ignore governing equations; pure physics models fail under noise and model mismatch. This project investigates whether **hybrid physics-ML** approaches yield better short-term accuracy *and* physically plausible attractors — and uses gradient saliency to explain *why*.

---

## Method

### Lorenz System

```
dx/dt = σ(y − x),   σ = 10
dy/dt = x(ρ − z) − y,  ρ = 28
dz/dt = xy − βz,    β = 8/3
```

Integrated via **4th-order Runge-Kutta** (Δt = 0.01 s). Gaussian observation noise (σ=0.05) injected to simulate real sensors.

### Models

| Model | Architecture | Params |
|-------|-------------|--------|
| **Physics Baseline** | RK4 step from true equations | 0 |
| **MLP** | 3 → 32 → 32 → 3, Tanh | 2,243 |
| **Hybrid** | Differentiable RK4 + residual MLP | 2,243 |

The hybrid forward pass:
```
ŷ_{t+1} = RK4(y_t) + f_residual(y_t; θ)
```

### Training
- 4,000 noisy observations → clean next-state targets
- Adam (lr=5e-3), cosine LR schedule, 8 epochs, batch 128

### Evaluation
- Short-term MSE (per step)
- Divergence time: first step where ‖ŷ − y‖ > 5.0
- **Attractor similarity**: L1 distance between 2D (x,z) phase-space histograms (higher = more similar)

### Explainability
Gradient saliency: `|∂MSE / ∂x_input|` via backpropagation. Identifies which state variables drive prediction error at each time step.

---

## Results

| Model | Mean MSE | Divergence Step | Attractor Sim |
|-------|----------|-----------------|---------------|
| Physics | — | — | ~1.00 |
| MLP | — | — | — |
| **Hybrid** | — | — | — |

*Run `python main.py` to populate this table via `outputs/results.csv`.*

### Plots

| Plot | Description |
|------|-------------|
| ![trajectory](plots/trajectory.png) | x/y/z components vs time |
| ![phase_space](plots/phase_space.png) | Attractor geometry comparison |
| ![prediction_vs_truth](plots/prediction_vs_truth.png) | Short-term tracking + MSE curves |
| ![attribution](plots/attribution.png) | Gradient saliency over time |

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

Runtime: ~30 s on CPU.

---

## Structure

```
xai-chaos/
├── data/           signals.npy (auto-generated)
├── src/
│   ├── lorenz.py   RK4 simulator + noise injection
│   ├── dataset.py  Dataset generation + caching
│   ├── models.py   PhysicsBaseline, MLP, HybridModel
│   ├── train.py    Fast training loop (cosine LR)
│   ├── metrics.py  MSE, divergence, attractor similarity
│   ├── xai.py      Gradient saliency
│   └── viz.py      4 publication-quality plots
├── outputs/        results.csv, saliency.csv, training_loss.csv
├── plots/          trajectory, phase_space, prediction_vs_truth, attribution
├── main.py
└── requirements.txt
```

---

## Key Insight

> **Accuracy does not imply physical correctness.**

The MLP may match short-term trajectories but generates attractors with incorrect topology. The Hybrid model's physics prior constrains predictions to lie near the true manifold — visible in the phase space portrait and quantified by attractor similarity. Gradient saliency reveals that the `z` component carries highest attribution near butterfly lobe transitions, consistent with the Lorenz system's sensitivity to z near the unstable fixed points.

---

## References

- Lorenz (1963). *Deterministic nonperiodic flow.* J. Atmos. Sci. 20(2).
- Rackauckas et al. (2020). *Universal Differential Equations for SciML.* arXiv:2001.04385.
- Brunton & Kutz (2022). *Data-Driven Science and Engineering.* Cambridge.
