"""Explicit least-squares straight-line fit (normal equations), as covered
in the course's curve-fitting unit. Used here to extract growth rates,
empirical integrator order, and power-law exponents from simulation data.
"""
import numpy as np


def least_squares_line(x, y):
    """
    Fit y = slope*x + intercept by ordinary least squares (normal equations).
    Returns (slope, intercept, r_squared).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xbar, ybar = x.mean(), y.mean()
    Sxy = np.sum((x - xbar) * (y - ybar))
    Sxx = np.sum((x - xbar) ** 2)
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - ybar) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2
