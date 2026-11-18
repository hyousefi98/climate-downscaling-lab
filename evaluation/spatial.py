"""FSS conventions are explicit; historical table and figure variants differ."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def exceedance(values, threshold):
    """Compare in float64 so a fitted quantile is not rounded to a float32 datum."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    a = np.asarray(values)
    return np.where(np.isfinite(a), a.astype(np.float64) >= float(threshold), np.nan)


def fractions(field, mask, window, support="valid", centers="expanded"):
    """Neighborhood mean of binary/probability fields; spatial axes are last two.

    support='valid' divides by valid cells, including truncated domain boundaries.
    support='zero_padded' treats masked/outside cells as zeros over a full window,
    reproducing the older table scorer. centers='expanded' retains neighborhoods
    containing any valid cell (historical figures); 'land' scores only mask centers.
    """
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 != 1:
        raise ValueError("Window must be a positive odd integer")
    a = np.asarray(field, dtype=np.float64)
    if a.ndim < 2 or support not in ("valid", "zero_padded") or centers not in ("expanded", "land"):
        raise ValueError("Invalid FSS shape or support/center policy")
    valid = np.broadcast_to(np.asarray(mask, bool), a.shape)
    if not np.isfinite(a[valid]).all():
        raise ValueError("Nonfinite values on declared spatial support")
    if ((a[valid] < 0) | (a[valid] > 1)).any():
        raise ValueError("Fractions input must be binary events or probabilities")
    size = (1,) * (a.ndim - 2) + (window, window)
    numerator = uniform_filter(np.where(valid, a, 0), size=size, mode="constant", cval=0)
    denominator = uniform_filter(valid.astype(float), size=size, mode="constant", cval=0)
    # A real nonempty neighborhood has at least one cell, hence mass >= 1/w².
    # Running-sum roundoff can leave tiny positive remnants over empty regions.
    has_support = denominator > (0.5 / window**2)
    if support == "valid":
        f = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=has_support)
        f = np.clip(f, 0, 1)
        eligible = has_support
    else:
        f = numerator
        eligible = np.ones_like(valid)
    if centers == "land":
        eligible &= valid
    return f, eligible


def fss(reference, prediction, threshold, window, mask=None, *, prediction_threshold=None,
        ensemble_probability=False, support="valid", centers="expanded", temporal="pooled"):
    """FSS for deterministic fields or ensemble exceedance probabilities.

    Pooled time-space numerator/denominator matches main FSS figures. mean_time
    averages defined per-time ratios, matching the older table scorer. All-dry
    cases are undefined, not perfect. Each product may have a distinct frozen
    training quantile, passed through prediction_threshold.
    """
    o = np.asarray(reference)
    p = np.asarray(prediction)
    if o.ndim == 2:
        o = o[None]
        p = p[:, None] if ensemble_probability else p[None]
    if o.ndim != 3 or temporal not in ("pooled", "mean_time"):
        raise ValueError("Reference must be (time,y,x) and temporal policy explicit")
    valid = np.isfinite(o) if mask is None else np.broadcast_to(mask, o.shape)
    if not np.isfinite(o[valid]).all():
        raise ValueError("Invalid reference on spatial support")
    pt = threshold if prediction_threshold is None else prediction_threshold
    if ensemble_probability:
        if p.ndim != 4 or p.shape[1:] != o.shape or p.shape[0] == 0 or not np.isfinite(p[:, valid]).all():
            raise ValueError("All ensemble members must match the fixed reference support")
        event_p = exceedance(p, pt).mean(axis=0)
    else:
        if p.shape != o.shape or not np.isfinite(p[valid]).all():
            raise ValueError("Prediction must match reference and be finite on support")
        event_p = exceedance(p, pt)
    a, va = fractions(exceedance(o, threshold), valid, window, support, centers)
    b, vb = fractions(event_p, valid, window, support, centers)
    use = va & vb
    num = np.where(use, (a-b)**2, 0).sum(axis=(1,2))
    den = np.where(use, a*a+b*b, 0).sum(axis=(1,2))
    if temporal == "pooled":
        return float(1 - num.sum()/den.sum()) if den.sum() > 0 else None
    ok = den > 0
    return float(np.mean(1 - num[ok]/den[ok])) if ok.any() else None
