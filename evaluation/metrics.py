"""Array metrics; inputs must already have matching physical units and support.

PBIAS uses sum(abs(reference)), as in the paper's archived scorers. This matters
for signed wind components. Positive bias denotes prediction minus reference.
No precipitation clipping or threshold is applied implicitly. Missing predictions
on a declared reference mask are errors, never silently removed to improve scores.
"""
from __future__ import annotations

import numpy as np


def paired_values(reference, prediction, mask=None):
    obs, pred = np.asarray(reference), np.asarray(prediction)
    if obs.shape != pred.shape:
        raise ValueError("Reference and prediction shapes differ")
    support = np.isfinite(obs) if mask is None else np.broadcast_to(np.asarray(mask, bool), obs.shape)
    if not np.any(support):
        raise ValueError("Empty verification support")
    if not np.isfinite(obs[support]).all() or not np.isfinite(pred[support]).all():
        raise ValueError("Nonfinite values on the fixed verification support")
    return obs[support].astype(np.float64), pred[support].astype(np.float64)


def ensemble_mean(members, reference_mask=None, dtype=np.float32):
    """Mean across axis 0; float32 matches the archived CorrDiff writer/scorer."""
    a = np.asarray(members)
    if a.ndim < 2 or a.shape[0] < 1:
        raise ValueError("Expected a nonempty leading ensemble axis")
    support = np.ones(a.shape[1:], bool) if reference_mask is None else np.broadcast_to(reference_mask, a.shape[1:])
    if not np.isfinite(a[:, support]).all():
        raise ValueError("Every ensemble member must be finite on reference support")
    result = np.full(a.shape[1:], np.nan, dtype=dtype)
    result[support] = a[:, support].mean(axis=0, dtype=dtype)
    return result


class Accumulator:
    """Stable pooled moments, including covariance, for streaming NetCDF files."""
    def __init__(self):
        self.n = 0
        self.mo = self.mp = self.m2o = self.m2p = self.cross = 0.0
        self.sse = self.sae = self.error = self.abs_obs = 0.0

    def add(self, reference, prediction, mask=None):
        o, p = paired_values(reference, prediction, mask)
        n = o.size
        mo, mp = float(o.mean()), float(p.mean())
        do, dp = o - mo, p - mp
        total = self.n + n
        w = self.n * n / total
        delta_o, delta_p = mo - self.mo, mp - self.mp
        self.m2o += float(do @ do) + w * delta_o**2
        self.m2p += float(dp @ dp) + w * delta_p**2
        self.cross += float(do @ dp) + w * delta_o * delta_p
        self.mo += delta_o * n / total
        self.mp += delta_p * n / total
        self.n = total
        e = p - o
        self.sse += float(e @ e)
        self.sae += float(np.abs(e).sum())
        self.error += float(e.sum())
        self.abs_obs += float(np.abs(o).sum())

    def result(self):
        if not self.n:
            raise ValueError("No values accumulated")
        return {
            "count": self.n, "RMSE": float(np.sqrt(self.sse / self.n)),
            "MAE": self.sae / self.n,
            "PBIAS_percent": 100 * self.error / self.abs_obs if self.abs_obs else None,
            "Pearson_r": float(np.clip(self.cross / np.sqrt(self.m2o * self.m2p), -1, 1))
            if self.m2o > 0 and self.m2p > 0 else None,
        }


def basic_metrics(reference, prediction, mask=None):
    a = Accumulator()
    a.add(reference, prediction, mask)
    return a.result()


def csi(reference, prediction, threshold, mask=None):
    """Pooled CSI with >= exceedances; undefined (None) when neither has events."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    o, p = paired_values(reference, prediction, mask)
    hit = (o >= threshold) & (p >= threshold)
    union = (o >= threshold) | (p >= threshold)
    return float(hit.sum() / union.sum()) if union.any() else None


def crps(reference, members, mask=None):
    """Empirical ensemble CRPS, E|X-y| - 0.5 E|X-X'| (not fair CRPS)."""
    o = np.asarray(reference)
    a = np.asarray(members)
    if a.shape[1:] != o.shape or a.shape[0] < 1:
        raise ValueError("Expected members with shape (ensemble, *reference.shape)")
    support = np.isfinite(o) if mask is None else np.broadcast_to(mask, o.shape)
    paired_values(o, a[0], support)
    if not np.isfinite(a[:, support]).all():
        raise ValueError("Missing ensemble values on reference support")
    x = np.sort(a[:, support].astype(np.float64), axis=0)
    m = x.shape[0]
    coefficients = (2 * np.arange(1, m + 1) - m - 1)[:, None]
    scores = np.abs(x - o[support]).mean(axis=0) - (coefficients * x).sum(axis=0) / (m*m)
    return float(scores.mean())


def kge(reference, prediction, mask=None):
    """Original KGE for positive-mean precipitation, not signed-wind ranking.

    Uses correlation, SD ratio and mean ratio (not KGE 2012); returns None when
    the reference mean is nonpositive or either variance vanishes.
    """
    o, p = paired_values(reference, prediction, mask)
    if o.size < 2 or o.mean() <= 0 or o.std() <= 0 or p.std() <= 0:
        return None
    r = np.corrcoef(o, p)[0, 1]
    return float(1 - np.sqrt((r - 1)**2 + (p.std()/o.std() - 1)**2 + (p.mean()/o.mean() - 1)**2))


def select_oracle_member(reference, members, criterion="kge", mask=None):
    """Retrospective selection using truth: main case figures KGE, supplements RMSE.

    This is an illustrative oracle, not an operational forecast or ensemble score.
    Returns the zero-based member index, first member on a tie. Never changes masks.
    """
    a = np.asarray(members)
    if a.shape[1:] != np.shape(reference) or a.shape[0] < 1:
        raise ValueError("Ensemble shape differs from reference")
    if criterion not in ("kge", "rmse"):
        raise ValueError("Criterion must be kge or rmse")
    scores = [kge(reference, x, mask) if criterion == "kge" else basic_metrics(reference, x, mask)["RMSE"] for x in a]
    valid = np.array([s is not None and np.isfinite(s) for s in scores])
    if not valid.any():
        raise ValueError("Oracle criterion undefined for every member")
    worst = -np.inf if criterion == "kge" else np.inf
    values = np.array([s if ok else worst for s, ok in zip(scores, valid)])
    return int(np.argmax(values) if criterion == "kge" else np.argmin(values))
