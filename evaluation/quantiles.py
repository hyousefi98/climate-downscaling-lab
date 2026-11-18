"""Fit product-specific quantiles on training data, including valid dry values."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainingQuantile:
    probability: float
    threshold: float
    count: int
    zero_count: int
    product: str
    first_training_time: str
    last_training_time: str


def fit_training_quantile(values, times, probability, *, product, evaluation_start, mask=None):
    a = np.asarray(values)
    t = np.asarray(times, dtype="datetime64[ns]")
    split = np.datetime64(evaluation_start, "ns")
    if a.ndim < 1 or len(t) != a.shape[0] or len(t) == 0:
        raise ValueError("Training timestamps must match first data axis")
    if np.isnat(t).any() or np.isnat(split) or np.any(t >= split):
        raise ValueError("Quantile fitting must use only timestamps before evaluation")
    if len(np.unique(t)) != len(t) or not 0 <= probability <= 1 or not product:
        raise ValueError("Invalid training coordinates, probability or product identifier")
    use = np.isfinite(a) if mask is None else np.broadcast_to(mask, a.shape)
    if not use.any() or not np.isfinite(a[use]).all():
        raise ValueError("Quantile support is empty or contains invalid values")
    x = a[use].astype(np.float64)
    return TrainingQuantile(float(probability), float(np.quantile(x, probability, method="linear")),
                            x.size, int((x == 0).sum()), product, str(t.min()), str(t.max()))
