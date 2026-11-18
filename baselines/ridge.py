"""Multivariate ridge reference, with an unpenalized intercept.

The archived paper fit used physical-unit pixels, alpha=1e-6, shuffled training
images and finite masked samples. See historical/generate_multivariate_baseline_nc.py
for that exact sampling/loader version; it is deliberately not silently replaced.
"""
import numpy as np


def fit_ridge(x, y, alpha=1e-6):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("x and y must be nonempty sample-by-channel matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Nonfinite samples or invalid ridge penalty")
    a = np.column_stack([x, np.ones(len(x))])
    gram = a.T @ a
    gram[:-1, :-1] += alpha * np.eye(x.shape[1])
    return np.linalg.solve(gram, a.T @ y)


def predict_ridge(x, weights):
    x, w = np.asarray(x, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1]+1 != w.shape[0] or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("Invalid feature/weight matrix")
    return x @ w[:-1] + w[-1]
