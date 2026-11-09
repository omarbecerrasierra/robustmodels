"""Robust univariate location estimators.

Every function takes a one-dimensional sample of finite values and returns a
float; an empty sample gives ``nan``.
"""

import numpy as np

from ._validation import as_1d, check_count, check_option, check_range
from .scale import _bisquare_weights, _matrix_median, _pairwise_rows, _tau_estimates, mad

__all__ = ["hodges_lehmann", "m_location", "tau_location", "trimmed_mean", "winsorized_mean"]


def hodges_lehmann(x):
    """Hodges-Lehmann estimator: the median of the Walsh averages.

    Median of ``(x_i + x_j) / 2`` over ``i <= j``. Breakdown point 29.3%,
    Gaussian efficiency 95.5%. Computed exactly in O(n log n) time.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.

    Returns
    -------
    float
        ``nan`` for an empty sample.

    References
    ----------
    Hodges, J. L. and Lehmann, E. L. (1963). Estimates of location based on rank
    tests. *Annals of Mathematical Statistics*, 34(2), 598-611.

    Examples
    --------
    >>> from robustmodels.location import hodges_lehmann
    >>> hodges_lehmann([1, 2, 2, 3, 4, 100])
    3.0
    """
    x = np.sort(as_1d(x))
    if x.size == 0:
        return np.nan
    first, last = _pairwise_rows(x.size, diagonal=True)
    return _matrix_median(lambda i, j: (x[i] + x[j]) / 2, first, last)


def trimmed_mean(x, proportion=0.1):
    """Trimmed mean, as ``scipy.stats.trim_mean``.

    Mean after removing ``floor(proportion * n)`` observations from each end of
    the sorted sample. Breakdown point close to ``proportion``.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    proportion : float in [0, 0.5), default 0.1
        Fraction removed from each end.

    Returns
    -------
    float
        ``nan`` for an empty sample.

    Examples
    --------
    >>> from robustmodels.location import trimmed_mean
    >>> trimmed_mean([1, 2, 2, 3, 4, 100], proportion=0.2)
    2.75
    """
    x = np.sort(as_1d(x))
    count = _trim_count(x.size, proportion)
    if x.size == 0:
        return np.nan
    return float(x[count : x.size - count].mean())


def winsorized_mean(x, proportion=0.1):
    """Winsorized mean.

    Mean after replacing the ``k = floor(proportion * n)`` smallest observations
    by the ``(k + 1)``-th smallest, and the ``k`` largest by the ``(k + 1)``-th
    largest. Breakdown point close to ``proportion``.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    proportion : float in [0, 0.5), default 0.1
        Fraction clipped at each end.

    Returns
    -------
    float
        ``nan`` for an empty sample.

    Examples
    --------
    >>> from robustmodels.location import winsorized_mean
    >>> round(winsorized_mean([1, 2, 2, 3, 4, 100], proportion=0.2), 4)
    2.8333
    """
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
    """Tau location of Maronna & Zamar (2002): one bisquare-weighted mean.

    Weighted mean with weights ``(1 - (r_i / c1)^2)^2`` for ``|r_i| < c1`` and 0
    otherwise, where ``r_i = (x_i - median) / MAD0`` and ``MAD0`` is the raw
    (unscaled) MAD. This is the location used by
    :func:`robustmodels.scale.tau_scale`. Returns the median if ``MAD0`` is zero
    or every weight is zero.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    c1 : float, default 4.5
        Bisquare tuning constant.

    Returns
    -------
    float
        ``nan`` for an empty sample.

    References
    ----------
    Maronna, R. A. and Zamar, R. H. (2002). Robust estimates of location and
    dispersion for high-dimensional datasets. *Technometrics*, 44(4), 307-317.

    Examples
    --------
    >>> from robustmodels.location import tau_location
    >>> round(tau_location([1, 2, 2, 3, 4, 100]), 4)
    2.3918
    """
    x = as_1d(x)
    check_range("c1", c1, 0, np.inf, low_open=True)
    if x.size == 0:
        return np.nan
    location, _ = _tau_estimates(x, c1)
    return location


PSI_C = {"huber": 1.345, "bisquare": 4.685}


def m_location(x, *, psi="huber", c=None, scale=None, tol=1e-9, max_iter=500):
    """M-estimator of location with Huber's or Tukey's bisquare psi function.

    Solves ``sum(psi((x_i - mu) / s)) = 0`` for ``mu`` by iteration from the
    median, with the scale ``s`` held fixed. With the MAD as scale the breakdown
    point is 50%; the default tuning constants give 95% Gaussian efficiency.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    psi : {"huber", "bisquare"}, default "huber"
        Huber's monotone psi or Tukey's redescending bisquare psi.
    c : float, optional
        Tuning constant. Defaults to 1.345 for "huber" and 4.685 for "bisquare".
    scale : float, optional
        Fixed scale ``s``. Defaults to the MAD about the median. A zero scale
        returns the median.
    tol : float, default 1e-9
        Stop when the location changes by less than ``tol * s``.
    max_iter : int, default 500
        Maximum number of iterations.

    Returns
    -------
    float
        ``nan`` for an empty sample. If the bisquare gives every observation zero
        weight, the current location is returned.

    References
    ----------
    Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of
    Mathematical Statistics*, 35(1), 73-101.

    Beaton, A. E. and Tukey, J. W. (1974). The fitting of power series, meaning
    polynomials, illustrated on band-spectroscopic data. *Technometrics*, 16(2),
    147-185.

    Examples
    --------
    >>> from robustmodels.location import m_location
    >>> round(m_location([1, 2, 2, 3, 4, 100]), 4)
    2.7988
    >>> round(m_location([1, 2, 2, 3, 4, 100], psi="bisquare"), 4)
    2.3869
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
