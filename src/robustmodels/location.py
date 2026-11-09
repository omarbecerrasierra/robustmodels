"""Robust univariate location estimators."""

import numpy as np

from ._validation import as_1d, check_count, check_option, check_range
from .scale import _bisquare_weights, _matrix_median, _pairwise_rows, _tau_estimates, mad

__all__ = ["hodges_lehmann", "m_location", "tau_location", "trimmed_mean", "winsorized_mean"]


def hodges_lehmann(x):
    """Median of the Walsh averages (x[i] + x[j]) / 2, for i <= j.

    Breakdown 29.3%, Gaussian efficiency 95.5%. Exact in O(n log n).
    """
    x = np.sort(as_1d(x))
    if x.size == 0:
        return np.nan
    first, last = _pairwise_rows(x.size, diagonal=True)
    return _matrix_median(lambda i, j: (x[i] + x[j]) / 2, first, last)


def trimmed_mean(x, proportion=0.1):
    """Mean after removing floor(proportion * n) values from each end."""
    x = np.sort(as_1d(x))
    count = _trim_count(x.size, proportion)
    if x.size == 0:
        return np.nan
    return float(x[count : x.size - count].mean())


def winsorized_mean(x, proportion=0.1):
    """Mean after clipping floor(proportion * n) values at each end."""
    x = np.sort(as_1d(x))
    count = _trim_count(x.size, proportion)
    if x.size == 0:
        return np.nan
    clipped = np.clip(x, x[count], x[-1 - count])
    return float(clipped.mean())


def _trim_count(n, proportion):
    check_range("proportion", proportion, 0, 0.5, high_open=True)
    return int(proportion * n)


def tau_location(x, *, c1=4.5):
    """One bisquare-weighted mean around the median, using the raw MAD."""
    x = as_1d(x)
    check_range("c1", c1, 0, np.inf, low_open=True)
    if x.size == 0:
        return np.nan
    location, _ = _tau_estimates(x, c1)
    return location


PSI_C = {"huber": 1.345, "bisquare": 4.685}


def m_location(x, *, psi="huber", c=None, scale=None, tol=1e-9, max_iter=500):
    """Solve sum(psi((x - location) / scale)) = 0, starting at the median.

    ``psi`` is "huber" or "bisquare". Default tuning gives 95% Gaussian
    efficiency. Scale defaults to the MAD; zero scale returns the median.
    If bisquare gives every observation zero weight, keep the current location.
    """
    x = as_1d(x)
    check_option("psi", psi, PSI_C)
    c = PSI_C[psi] if c is None else c
    _check_parameters(c, scale, tol, max_iter)
    if x.size == 0:
        return np.nan
    location = float(np.median(x))
    scale = mad(x, center=location) if scale is None else scale
    if scale == 0:
        return location
    for _ in range(max_iter):
        updated = _location_step(x, location, scale, psi, c)
        if abs(updated - location) < tol * scale:
            return float(updated)
        location = updated
    return float(location)


def _check_parameters(c, scale, tol, max_iter):
    check_range("c", c, 0, np.inf, low_open=True)
    check_range("tol", tol, 0, np.inf, low_open=True)
    check_count("max_iter", max_iter)
    if scale is not None:
        check_range("scale", scale, 0, np.inf)


def _location_step(x, location, scale, psi, c):
    if psi == "huber":
        clipped = np.clip(x, location - c * scale, location + c * scale)
        return float(clipped.mean())
    weights = _bisquare_weights((x - location) / scale, c)
    total = weights.sum()
    if total == 0:
        return location
    return float(np.dot(weights, x) / total)
