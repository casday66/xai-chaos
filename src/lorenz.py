"""Lorenz-63 simulator: RK4 integration + optional noise injection."""
import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def _deriv(s):
    x, y, z = s
    return np.array([SIGMA * (y - x),
                     x * (RHO - z) - y,
                     x * y - BETA * z])


def rk4_step(s, dt=0.01):
    k1 = _deriv(s)
    k2 = _deriv(s + .5 * dt * k1)
    k3 = _deriv(s + .5 * dt * k2)
    k4 = _deriv(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(n_steps, dt=0.01, ic=None, noise_std=0.0, seed=0):
    """Return trajectory (n_steps+1, 3). Add Gaussian obs noise if noise_std>0."""
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 0.0, 0.0]) if ic is None else np.asarray(ic, float)
    traj = np.empty((n_steps + 1, 3))
    traj[0] = s
    for i in range(n_steps):
        s = rk4_step(s, dt)
        traj[i + 1] = s
    if noise_std > 0:
        traj += rng.normal(0, noise_std, traj.shape)
    return traj
