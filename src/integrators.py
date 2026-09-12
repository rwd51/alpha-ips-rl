"""
Numerical time-integration of the idealized gradient-flow ODE z_dot = f(z).

Implements the two schemes required by the course (Chapra & Canale ch. 25):
  - Euler's method            (25.1)  -- O(h) global error
  - Classical 4th-order RK4   (25.3)  -- O(h^4) global error
applied to the *system* of K coupled logit equations (25.4: systems of ODEs).

Both operate on a batch of independent seeds simultaneously: z has shape
(S, K) and f(z) must return an array of the same shape.
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def euler_step(f: Callable[[np.ndarray], np.ndarray], z: np.ndarray, h: float) -> np.ndarray:
    return z + h * f(z)


def rk4_step(f: Callable[[np.ndarray], np.ndarray], z: np.ndarray, h: float) -> np.ndarray:
    k1 = f(z)
    k2 = f(z + 0.5 * h * k1)
    k3 = f(z + 0.5 * h * k2)
    k4 = f(z + h * k3)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


_STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def integrate(f: Callable[[np.ndarray], np.ndarray], z0: np.ndarray, h: float,
              n_steps: int, method: str = "rk4", record_every: int = 1):
    """
    Integrate z_dot = f(z) from z0 for n_steps of size h.

    Returns
    -------
    ts   : (n_recorded+1,) time stamps (includes t=0)
    traj : (n_recorded+1, S, K) trajectory of z (includes z0)
    """
    step = _STEPPERS[method]
    z = np.array(z0, dtype=np.float64, copy=True)
    ts = [0.0]
    traj = [z.copy()]
    t = 0.0
    for i in range(n_steps):
        z = step(f, z, h)
        t += h
        if (i + 1) % record_every == 0:
            ts.append(t)
            traj.append(z.copy())
    return np.array(ts), np.array(traj)


def time_to_event(f: Callable[[np.ndarray], np.ndarray], z0: np.ndarray, h: float,
                   event: Callable[[np.ndarray], bool], method: str = "rk4",
                   max_steps: int = 2_000_000):
    """
    Step-and-check event detection: integrate z_dot = f(z) from z0 until
    event(z) becomes True (e.g. max softmax probability crosses a collapse
    threshold), without wasting compute recording the full trajectory.
    Returns the event time, or None if the budget is exhausted first.
    z0 must be a single (1, K) or (K,) state (not a batch).
    """
    step = _STEPPERS[method]
    z = np.array(z0, dtype=np.float64, copy=True)
    t = 0.0
    if event(z):
        return t
    for _ in range(max_steps):
        z = step(f, z, h)
        t += h
        if event(z):
            return t
    return None


def integrate_at(f: Callable[[np.ndarray], np.ndarray], z0: np.ndarray, h: float,
                 record_steps, method: str = "rk4"):
    """
    Integrate z_dot = f(z) from z0 with fixed step h, recording z only at the
    given step indices (0 = z0) -- e.g. log-spaced indices, to plot a
    trajectory on a log time axis across many decades without storing every
    step.

    Returns
    -------
    ts   : (n_recorded,) time stamps
    traj : (n_recorded, S, K) trajectory of z
    """
    step = _STEPPERS[method]
    record_steps = np.unique(np.asarray(record_steps, dtype=np.int64))
    z = np.array(z0, dtype=np.float64, copy=True)
    traj = []
    n = 0
    for target in record_steps:
        while n < target:
            z = step(f, z, h)
            n += 1
        traj.append(z.copy())
    return record_steps * h, np.array(traj)
