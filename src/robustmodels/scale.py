"""Robust scale, skewness and their numerical calculations.

Every public function takes a one-dimensional sample of finite values and
returns a float. An empty sample gives ``nan`` and a single observation has
zero scale.
"""

import numpy as np
from scipy import optimize, special
from scipy.stats import chi2, norm

from ._validation import as_1d, check_range

__all__ = ["lmc", "m_scale", "mad", "medcouple", "pn", "qn", "rmc", "sn", "tau_scale"]


MAD_CONSTANT = 1 / norm.ppf(0.75)


# MAD factors for n = 2, ..., 9; Croux & Rousseeuw (1992).
_MAD_N = (1.196, 1.495, 1.363, 1.206, 1.200, 1.140, 1.129, 1.107)


def mad(x, *, center=None, consistency=True, finite_correction=False):
    """Median absolute deviation, ``1.4826 * med|x_i - center|``.

    Breakdown point 50%, Gaussian efficiency 37%.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    center : float, optional
        Center of the deviations. Defaults to the sample median.
    consistency : bool, default True
        Multiply by ``1 / Phi^-1(3/4) = 1.4826``, so that the MAD estimates the
        standard deviation at the normal model.
    finite_correction : bool, default False
        Multiply by the small-sample factors of Croux & Rousseeuw (1992). Without
        them the MAD underestimates the normal standard deviation by 17% at n = 5.

    Returns
    -------
    float
        ``nan`` for an empty sample, 0 for a single observation.

    References
    ----------
    Hampel, F. R. (1974). The influence curve and its role in robust estimation.
    *Journal of the American Statistical Association*, 69(346), 383-393.

    Croux, C. and Rousseeuw, P. J. (1992). Time-efficient algorithms for two
    highly robust estimators of scale. *Computational Statistics*, 1, 411-428.

    Examples
    --------
    >>> from robustmodels.scale import mad
    >>> round(mad([1, 2, 2, 3, 4, 100]), 4)
    1.4826
    >>> mad([1, 2, 2, 3, 4, 100], consistency=False)
    1.0
    """
    x = as_1d(x)
    if center is not None:
        check_range("center", center, -np.inf, np.inf)
    if x.size < 2:
        return _trivial_scale(x)
    if center is None:
        center = np.median(x)
    value = np.median(np.abs(x - center))
    if consistency:
        value *= MAD_CONSTANT
    if finite_correction:
        value *= _mad_factor(x.size)
    return float(value)


def _mad_factor(n):
    return _MAD_N[n - 2] if n <= 9 else n / (n - 0.8)


QN_CONSTANT = 1 / (np.sqrt(2) * norm.ppf(5 / 8))


_QN_N = (
    0.399356,
    0.99365,
    0.51321,
    0.84401,
    0.6122,
    0.85877,
    0.66993,
    0.87344,
    0.72014,
    0.88906,
    0.75743,
)


def qn(x, *, consistency=True, finite_correction=True):
    """Qn scale estimator of Rousseeuw & Croux (1993).

    The ``k``-th smallest of the distances ``|x_i - x_j|``, ``i < j``, with
    ``k = h(h - 1) / 2`` and ``h = n // 2 + 1``. Breakdown point 50%, Gaussian
    asymptotic efficiency 82%. The current sorted-row selection uses
    O(n log^2 n) time and O(n) auxiliary storage, without forming all pairs.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    consistency : bool, default True
        Multiply by ``1 / (sqrt(2) * Phi^-1(5/8)) = 2.2191``, so that Qn estimates
        the standard deviation at the normal model.
    finite_correction : bool, default True
        Multiply by the small-sample factors used by R's robustbase, which make Qn
        approximately unbiased at the normal model for small ``n``.

    Returns
    -------
    float
        ``nan`` for an empty sample, 0 for a single observation.

    References
    ----------
    Rousseeuw, P. J. and Croux, C. (1993). Alternatives to the median absolute
    deviation. *Journal of the American Statistical Association*, 88(424), 1273-1283.

    Examples
    --------
    >>> from robustmodels.scale import qn
    >>> round(qn([1, 2, 2, 3, 4, 100]), 4)
    1.3586
    """
    x = np.sort(as_1d(x))
    if x.size < 2:
        return _trivial_scale(x)
    h = x.size // 2 + 1
    rank = h * (h - 1) // 2
    first, last = _pairwise_rows(x.size)
    value = _kth_smallest(lambda i, j: x[j] - x[i], first, last, rank)
    if consistency:
        value *= QN_CONSTANT
    if finite_correction:
        value *= _qn_factor(x.size)
    return float(value)


def _qn_factor(n):
    if n <= 12:
        return _QN_N[n - 2]
    if n % 2:
        correction = -2.1284 - 5.172 / n
        correction = 1.60188 + correction / n
    else:
        correction = 6.987 - 77 / n
        correction = 1.9654 + correction / n
        correction = 3.67561 + correction / n
    return 1 / (1 + correction / n)


SN_CONSTANT = 1.1925985531232086


# Sn factors for n = 2, ..., 9; Croux & Rousseeuw (1992).
_SN_N = (0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131)


def sn(x, *, consistency=True, finite_correction=True):
    """Sn scale estimator of Rousseeuw & Croux (1993).

    ``lomed_i himed_j |x_i - x_j|``: for each observation, the high median of its
    distances to all observations; then the low median of these values.
    Breakdown point 50%, Gaussian efficiency 58%. Computed exactly in
    O(n log n) time.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    consistency : bool, default True
        Multiply by 1.1926, so that Sn estimates the standard deviation at the
        normal model.
    finite_correction : bool, default True
        Multiply by the small-sample factors of Croux & Rousseeuw (1992).

    Returns
    -------
    float
        ``nan`` for an empty sample, 0 for a single observation.

    References
    ----------
    Rousseeuw, P. J. and Croux, C. (1993). Alternatives to the median absolute
    deviation. *Journal of the American Statistical Association*, 88(424), 1273-1283.

    Croux, C. and Rousseeuw, P. J. (1992). Time-efficient algorithms for two
    highly robust estimators of scale. *Computational Statistics*, 1, 411-428.

    Examples
    --------
    >>> from robustmodels.scale import sn
    >>> round(sn([1, 2, 2, 3, 4, 100]), 4)
    1.1843
    """
    x = np.sort(as_1d(x))
    if x.size < 2:
        return _trivial_scale(x)
    distances = np.sort(_himed_distances(x))
    lower_middle = (x.size + 1) // 2 - 1
    value = distances[lower_middle]
    if consistency:
        value *= SN_CONSTANT
    if finite_correction:
        value *= _sn_factor(x.size)
    return float(value)


def _sn_factor(n):
    if n <= 9:
        return _SN_N[n - 2]
    return n / (n - 0.9) if n % 2 else 1.0


def _himed_distances(x):
    """For sorted x and every i, the ``(n//2 + 1)``-th smallest ``|x_i - x_j|`` over j.

    The zero self-distance comes first, so this is the ``n//2``-th smallest of the
    merged sorted lists ``left[a] = x_i - x_(i-1-a)`` and ``right[b] = x_(i+1+b) - x_i``;
    a vectorized binary search finds how many come from the left list.
    """
    n, k = x.size, x.size // 2
    i = np.arange(n)

    def left(a):  # +inf past the end
        indices = np.clip(i - 1 - a, 0, n - 1)
        distances = x[i] - x[indices]
        return np.where(a < i, distances, np.inf)

    def right(b):  # -inf before the start
        indices = np.clip(i + 1 + b, 0, n - 1)
        distances = x[indices] - x[i]
        return np.where(b >= 0, distances, -np.inf)

    lo, hi = np.maximum(0, k - (n - 1 - i)), np.minimum(k, i)
    while np.any(lo < hi):
        active = lo < hi
        mid = (lo + hi) // 2
        more = left(mid) < right(k - mid - 1)
        lo[active & more] = mid[active & more] + 1
        hi[active & ~more] = mid[active & ~more]
    last_left = np.where(lo > 0, left(lo - 1), -np.inf)
    last_right = right(k - lo - 1)
    return np.maximum(last_left, last_right)


PN_CONSTANT = 1 / (np.sqrt(2) * norm.ppf(0.75))


# Tarr's reference implementation: factors for n = 3, ..., 40.
# https://www.maths.usyd.edu.au/u/gartht/R/Pn.R
_PN_N = (
    1.128,
    1.303,
    1.109,
    1.064,
    1.166,
    1.103,
    1.087,
    1.105,
    1.047,
    1.063,
    1.057,
    1.040,
    1.061,
    1.047,
    1.043,
    1.048,
    1.031,
    1.037,
    1.035,
    1.028,
    1.036,
    1.030,
    1.029,
    1.032,
    1.023,
    1.025,
    1.024,
    1.021,
    1.026,
    1.022,
    1.021,
    1.023,
    1.018,
    1.020,
    1.019,
    1.017,
    1.020,
    1.018,
)


def pn(x, *, consistency=True, finite_correction=True):
    """Pn scale estimator of Tarr, Müller & Weber (2012).

    Interquartile range of the pairwise means: ``H^-1(3/4) - H^-1(1/4)`` with
    ``H`` the empirical distribution of ``(x_i + x_j) / 2``, ``i < j``.
    Asymptotic breakdown point 13.4%, Gaussian efficiency 86%. The current
    sorted-row selection uses O(n log^2 n) time and O(n) auxiliary storage.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    consistency : bool, default True
        Multiply by ``1 / (sqrt(2) * Phi^-1(3/4)) = 1.0484``, so that Pn estimates
        the standard deviation at the normal model.
    finite_correction : bool, default True
        Multiply by the factors in Tarr's reference R code for ``n <= 40`` and
        by ``n / (n - 0.7)`` for larger samples.

    Returns
    -------
    float
        ``nan`` for an empty sample, 0 for one or two observations.

    References
    ----------
    Tarr, G., Müller, S. and Weber, N. C. (2012). A robust scale estimator based
    on pairwise means. *Journal of Nonparametric Statistics*, 24(1), 187-199.

    Tarr, G. (2014). *Quantile Based Estimation of Scale and Dependence*.
    PhD thesis, University of Sydney. https://hdl.handle.net/2123/10590

    Tarr's reference R implementation and finite-sample factors:
    https://www.maths.usyd.edu.au/u/gartht/R/Pn.R

    Examples
    --------
    One outlier in six exceeds the breakdown point of Pn:

    >>> from robustmodels.scale import pn
    >>> round(pn([1, 2, 2, 3, 4, 100]), 4)
    54.6572
    """
    x = np.sort(as_1d(x))
    if x.size < 3:
        return _trivial_scale(x)
    q1 = _pairwise_quantile(x, 0.25)
    q3 = _pairwise_quantile(x, 0.75)
    value = q3 - q1
    if consistency:
        value *= PN_CONSTANT
    if finite_correction:
        value *= _pn_factor(x.size)
    return float(value)


def _pn_factor(n):
    if n <= 40:
        return _PN_N[n - 3]
    return n / (n - 0.7)


def tau_scale(x, *, c1=4.5, c2=3.0, consistency=True):
    """Tau scale of Yohai & Zamar (1988), in the form of Maronna & Zamar (2002).

    ``s0 * sqrt(mean(min(((x - mu) / s0)^2, c2^2)) / E)``, with ``s0`` the raw
    (unscaled) MAD and ``mu`` the tau location
    (:func:`robustmodels.location.tau_location`).

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    c1 : float, default 4.5
        Bisquare tuning constant of the location weights.
    c2 : float, default 3.0
        Truncation point of the standardized residuals.
    consistency : bool, default True
        Divide by ``E = E[min(Z^2, (c2 * Phi^-1(3/4))^2)]``, so that the result
        estimates the standard deviation at the normal model.

    Returns
    -------
    float
        ``nan`` for an empty sample, 0 for a single observation or a zero MAD.

    References
    ----------
    Yohai, V. J. and Zamar, R. H. (1988). High breakdown-point estimates of
    regression by means of the minimization of an efficient scale. *Journal of
    the American Statistical Association*, 83(402), 406-413.

    Maronna, R. A. and Zamar, R. H. (2002). Robust estimates of location and
    dispersion for high-dimensional datasets. *Technometrics*, 44(4), 307-317.

    Examples
    --------
    >>> from robustmodels.scale import tau_scale
    >>> round(tau_scale([1, 2, 2, 3, 4, 100]), 4)
    1.5998
    """
    x = as_1d(x)
    check_range("c1", c1, 0, np.inf, low_open=True)
    check_range("c2", c2, 0, np.inf, low_open=True)
    if x.size < 2:
        return _trivial_scale(x)
    location, s0 = _tau_estimates(x, c1)
    if s0 == 0:
        return 0.0
    residuals = (x - location) / s0
    clipped_squares = np.minimum(residuals**2, c2**2)
    mean_square = clipped_squares.mean()
    if consistency:
        mean_square /= _e_trunc_square(c2 * norm.ppf(0.75))
    return float(s0 * np.sqrt(mean_square))


def _tau_estimates(x, c):
    """Return the tau location and raw MAD of a nonempty, finite sample."""
    center = np.median(x)
    raw_mad = np.median(np.abs(x - center))
    if raw_mad == 0:
        return float(center), 0.0
    residuals = (x - center) / raw_mad
    weights = _bisquare_weights(residuals, c)
    total = weights.sum()
    if total == 0:
        return float(center), float(raw_mad)
    location = np.dot(weights, x) / total
    return float(location), float(raw_mad)


def _e_trunc_square(b):
    """E[min(Z^2, b^2)] for Z ~ N(0, 1)."""
    return 2 * norm.cdf(b) - 1 - 2 * b * norm.pdf(b) + 2 * b * b * norm.sf(b)


def m_scale(x, *, center=None, breakdown=0.5):
    """Bisquare M-estimator of scale.

    Solves ``mean(rho((x_i - center) / s)) = breakdown`` for ``s``, with Tukey's
    bisquare ``rho`` scaled to a maximum of 1. Its tuning constant makes ``s``
    estimate the standard deviation at the normal model. This is the scale that
    S-estimators minimize.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of finite values.
    center : float, optional
        Center of the residuals. Defaults to the sample median; pass 0 for
        residuals that are already centered.
    breakdown : float in (0, 0.5], default 0.5
        Breakdown point, the right-hand side of the equation.

    Returns
    -------
    float
        ``nan`` for an empty sample. 0 for a single observation, or when at most a
        fraction ``breakdown`` of the residuals is nonzero.

    References
    ----------
    Maronna, R. A., Martin, R. D., Yohai, V. J. and Salibián-Barrera, M. (2019).
    *Robust Statistics: Theory and Methods (with R)*, 2nd ed. Wiley.

    Examples
    --------
    >>> from robustmodels.scale import m_scale
    >>> round(m_scale([1, 2, 2, 3, 4, 100]), 4)
    1.5019
    """
    x = as_1d(x)
    check_range("breakdown", breakdown, 0, 0.5, low_open=True)
    if center is not None:
        check_range("center", center, -np.inf, np.inf)
    if x.size < 2:
        return _trivial_scale(x)
    center = np.median(x) if center is None else center
    residuals = x - center
    tuning = _tuning_for_breakdown(breakdown)
    return _mscale_bisquare(residuals, tuning, breakdown)


def _mscale_bisquare(residuals, c, breakdown, s0=None, tol=1e-10, max_iter=200):
    """Solve mean(rho(residuals / scale)) = breakdown by fixed-point iteration."""
    residuals = np.abs(residuals)
    positive = residuals[residuals > 0]
    if positive.size <= breakdown * residuals.size:
        return 0.0
    scale = np.median(residuals) / 0.6745 if s0 is None else s0
    if scale <= 0:
        scale = np.median(positive) / 0.6745
    for _ in range(max_iter):
        if scale <= 0:
            return 0.0
        average_loss = np.mean(_rho(residuals / scale, c))
        updated = scale * np.sqrt(average_loss / breakdown)
        if abs(updated - scale) <= tol * scale:
            return float(updated)
        scale = updated

    # Near an exact fit the fixed-point iteration can be arbitrarily slow.
    # Finish with a bracketed root in log-scale, retaining relative accuracy
    # across units and without accepting an unconverged iterate.
    def equation(log_scale):
        return np.mean(_rho(residuals / np.exp(log_scale), c)) - breakdown

    lower = upper = np.log(scale)
    step = np.log(2.0)
    while equation(lower) < 0:
        lower -= step
    while equation(upper) > 0:
        upper += step
    root = optimize.brentq(equation, lower, upper, xtol=tol, rtol=1e-14)
    return float(np.exp(root))


def _rho(t, c):
    """Tukey's bisquare loss, normalized to rho(infinity) = 1."""
    squared = (t / c) ** 2
    clipped = np.minimum(squared, 1)
    return clipped * (3 + clipped * (clipped - 3))


def _bisquare_weights(t, c):
    """``psi(t) / t`` up to a constant."""
    u = (t / c) ** 2
    return np.where(u < 1, (1 - u) ** 2, 0.0)


def _expected_rho(c, p=1):
    """``E[rho_c(||Z||)]`` for ``Z ~ N_p(0, I)``."""
    t = c * c
    m1 = p * chi2.cdf(t, p + 2)
    m2 = p * (p + 2) * chi2.cdf(t, p + 4)
    m3 = p * (p + 2) * (p + 4) * chi2.cdf(t, p + 6)
    return 3 * m1 / t - 3 * m2 / t**2 + m3 / t**3 + chi2.sf(t, p)


def _tuning_for_breakdown(breakdown=0.5, p=1):
    """``c`` with ``E[rho_c] = breakdown``: consistent scale with that breakdown point."""
    check_range("breakdown", breakdown, 0, 0.5, low_open=True)
    upper = 10 * np.sqrt((p + 4) / breakdown)  # E[rho_c] < breakdown there
    return optimize.brentq(lambda c: _expected_rho(c, p) - breakdown, 1e-3, upper, xtol=1e-14)


def _shape_efficiency(c, p):
    """Gaussian efficiency of the bisquare M-estimator of shape (Tyler 1982).

    ``1 / sigma1``, ``sigma1 = p(p+2) E[psi^2 d^2] / E[psi' d^2 + (p+1) psi d]^2``
    with ``psi(d) = d (1 - (d/c)^2)^2`` and ``d = ||Z||``.
    """

    def m(a):  # E[d^a; d <= c]
        g = np.exp(special.gammaln((p + a) / 2) - special.gammaln(p / 2))
        return g * 2 ** (a / 2) * chi2.cdf(c * c, p + a)

    psi2_d2 = m(4) - 4 * m(6) / c**2 + 6 * m(8) / c**4 - 4 * m(10) / c**6 + m(12) / c**8
    dpsi_d2 = m(2) - 6 * m(4) / c**2 + 5 * m(6) / c**4
    psi_d = m(2) - 2 * m(4) / c**2 + m(6) / c**4
    return (dpsi_d2 + (p + 1) * psi_d) ** 2 / (p * (p + 2) * psi2_d2)


def _tuning_for_shape_efficiency(efficiency, p):
    check_range("efficiency", efficiency, 0, 1, low_open=True, high_open=True)
    return optimize.brentq(lambda c: _shape_efficiency(c, p) - efficiency, 0.5, 1e3, xtol=1e-12)


def medcouple(x):
    """Medcouple, a robust measure of skewness (Brys, Hubert & Struyf 2004).

    Median of ``((x_i - m) - (m - x_j)) / (x_i - x_j)`` over ``x_i >= m >= x_j``,
    with ``m`` the sample median and the paper's kernel for ties at ``m``.
    Lies in [-1, 1]: positive for right skew, 0 for symmetric or constant
    samples, and ``medcouple(-x) == -medcouple(x)``. Breakdown point 25%.
    The current sorted-row selection uses O(n log^2 n) time and O(n) auxiliary
    storage. For an even number of kernels the two middle values are averaged;
    this can differ from robustbase with ties at the median.

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
    Brys, G., Hubert, M. and Struyf, A. (2004). A robust measure of skewness.
    *Journal of Computational and Graphical Statistics*, 13(4), 996-1017.

    Examples
    --------
    >>> from robustmodels.scale import medcouple
    >>> medcouple([1, 2, 2, 3, 4, 100])
    0.5
    """
    x = np.sort(as_1d(x))
    if x.size == 0:
        return np.nan
    center = np.median(x)
    upper = x[x >= center] - center
    lower = x[x <= center] - center

    def kernel(i, j):
        tie = (upper[i] == 0) & (lower[j] == 0)
        denominator = np.where(tie, 1, upper[i] - lower[j])
        ratio = (upper[i] + lower[j]) / denominator
        tie_value = np.sign(i + j - lower.size + 1)
        return np.where(tie, tie_value, ratio)

    first = np.zeros(upper.size, dtype=int)
    last = np.full(upper.size, lower.size - 1)
    return _matrix_median(kernel, first, last)


def lmc(x):
    """Left medcouple, a robust measure of left tail weight (Brys, Hubert & Struyf 2006).

    ``LMC = -medcouple(x[x <= median(x)])``; larger values mean a heavier left tail.

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
    Brys, G., Hubert, M. and Struyf, A. (2006). Robust measures of tail weight.
    *Computational Statistics & Data Analysis*, 50(3), 733-759.

    Examples
    --------
    >>> from robustmodels.scale import lmc
    >>> round(lmc([0.2, 0.5, 0.9, 1.1, 1.6, 2.4, 3.8, 7.5]), 4)
    0.0556
    """
    x = as_1d(x)
    if x.size == 0:
        return np.nan
    return 0.0 - medcouple(x[x <= np.median(x)])


def rmc(x):
    """Right medcouple, a robust measure of right tail weight (Brys, Hubert & Struyf 2006).

    ``RMC = medcouple(x[x >= median(x)])``; larger values mean a heavier right tail.

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
    Brys, G., Hubert, M. and Struyf, A. (2006). Robust measures of tail weight.
    *Computational Statistics & Data Analysis*, 50(3), 733-759.

    Examples
    --------
    >>> from robustmodels.scale import rmc
    >>> round(rmc([0.2, 0.5, 0.9, 1.1, 1.6, 2.4, 3.8, 7.5]), 4)
    0.2458
    """
    x = as_1d(x)
    if x.size == 0:
        return np.nan
    return medcouple(x[x >= np.median(x)])


def _trivial_scale(x):
    """Scale of fewer than two values: NaN if empty, else 0."""
    return np.nan if x.size == 0 else 0.0


# Select pairwise values without allocating the full n-by-n matrix.
_DIRECT_LIMIT = 50_000


def _pairwise_rows(n, diagonal=False):
    """Inclusive column bounds for pairs i < j, or i <= j with the diagonal."""
    row_count = n if diagonal else n - 1
    first = np.arange(row_count)
    if not diagonal:
        first += 1
    last = np.full(row_count, n - 1)
    return first, last


def _pairwise_quantile(x, probability, diagonal=False):
    """Inverse empirical CDF of (x[i] + x[j]) / 2 for sorted x."""
    first, last = _pairwise_rows(x.size, diagonal)
    count = int(np.sum(last - first + 1))
    rank = int(np.ceil(probability * count))
    return _kth_smallest(lambda i, j: (x[i] + x[j]) / 2, first, last, rank)


def _matrix_median(value, first, last):
    """Median of an implicit sorted-row matrix, averaging the two middle values."""
    count = int(np.sum(last - first + 1))
    rank = (count + 1) // 2
    lower = _kth_smallest(value, first, last, rank)
    if count % 2:
        return lower
    upper = _kth_smallest(value, first, last, rank + 1)
    return float((lower + upper) / 2)


def _kth_smallest(value, lo, hi, k):
    """Select rank k (1-based) from value(i, j), with lo[i] <= j <= hi[i]."""
    lo = np.array(lo, copy=True)
    hi = np.array(hi, copy=True)
    while True:
        sizes = np.maximum(hi - lo + 1, 0)
        if sizes.sum() <= max(_DIRECT_LIMIT, 4 * lo.size):
            return _direct(value, lo, sizes, k)
        rows = np.flatnonzero(sizes)
        midpoints = (lo[rows] + hi[rows]) // 2
        trial = _weighted_median(value(rows, midpoints), sizes[rows])
        below = _count(value, rows, lo[rows], hi[rows], trial, strict=True)
        through = _count(value, rows, lo[rows], hi[rows], trial, strict=False)
        if k <= below.sum():
            hi[rows] = lo[rows] + below - 1
        elif k > through.sum():
            k -= through.sum()
            lo[rows] += through
        else:
            return float(trial)


def _direct(value, first, sizes, rank):
    """Expand only the remaining candidates and select their order statistic."""
    rows = np.repeat(np.arange(first.size), sizes)
    row_offsets = np.cumsum(sizes) - sizes
    offsets = np.arange(rows.size) - np.repeat(row_offsets, sizes)
    columns = np.repeat(first, sizes) + offsets
    candidates = value(rows, columns)
    return float(np.partition(candidates, rank - 1)[rank - 1])


def _weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    middle = np.searchsorted(cumulative, cumulative[-1] / 2)
    return values[order[middle]]


def _count(value, rows, lo, hi, trial, strict):
    """Count entries below trial in each row, including ties if strict is False."""
    left = lo.copy()
    right = hi + 1
    while np.any(left < right):
        active = left < right
        middle = np.minimum((left + right) // 2, hi)
        values = value(rows, middle)
        below = values < trial if strict else values <= trial
        move_left = active & below
        move_right = active & ~below
        left[move_left] = middle[move_left] + 1
        right[move_right] = middle[move_right]
    return left - lo
