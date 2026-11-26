"""Shape-constrained estimation: log-concave densities and antitonic score matching.

:class:`LogConcaveDensity` is the nonparametric maximum likelihood estimate of
a univariate log-concave density, which needs no bandwidth.
:class:`ScoreMatchingRegressor` is a linear regression whose convex loss is
estimated from the data by antitonic score matching.
"""

import numpy as np
from scipy import integrate, linalg, signal, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import isotonic_regression
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_1d, is_exact_fit, standardize_columns
from .rank import _l1_fit
from .regression import _coefficient_table
from .scale import mad

__all__ = ["LogConcaveDensity", "ScoreMatchingRegressor"]


class LogConcaveDensity(BaseEstimator):
    """Maximum likelihood estimate of a univariate log-concave density (Walther 2002).

    Among all densities ``f = exp(phi)`` with ``phi`` concave, which include
    the normal, logistic, gamma and beta densities with shape at least one,
    the estimate maximizes the likelihood. It exists and is unique for two or
    more distinct observations, needs no bandwidth, and ``phi`` is piecewise
    linear with knots at some observations, ``-inf`` outside their range
    (Pal, Woodroofe & Meyer 2007; Dümbgen & Rufibach 2009). It is computed by
    the active set algorithm of Dümbgen, Hüsler & Rufibach (2007), as R's
    ``logcondens``: Newton steps for the values of ``phi`` at a set of knots,
    which grows where the likelihood can increase and shrinks where ``phi``
    would stop being concave. The estimate has the mean of the sample and at
    most its variance (divisor ``n``).

    Attributes
    ----------
    support_ : ndarray of shape (n_distinct,)
        Sorted distinct observations.
    weights_ : ndarray of shape (n_distinct,)
        Share of the sample at each of them.
    log_density_ : ndarray of shape (n_distinct,)
        ``phi`` at ``support_``.
    knots_ : ndarray
        Observations where the slope of ``phi`` changes, with the extremes.

    References
    ----------
    Walther, G. (2002). Detecting the presence of mixing with multiscale
    maximum likelihood. *Journal of the American Statistical Association*,
    97(458), 508-513.

    Pal, J. K., Woodroofe, M. and Meyer, M. (2007). Estimating a Polya
    frequency function_2. In *Complex Datasets and Inverse Problems*, IMS
    Lecture Notes 54, 239-249.

    Dümbgen, L. and Rufibach, K. (2009). Maximum likelihood estimation of a
    log-concave density and its distribution function: basic properties and
    uniform consistency. *Bernoulli*, 15(1), 40-68.

    Dümbgen, L., Hüsler, A. and Rufibach, K. (2007). Active set and EM
    algorithms for log-concave densities based on complete and censored data.
    Technical report 61, University of Bern. arXiv:0707.4643.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.shape import LogConcaveDensity
    >>> x = np.random.default_rng(0).gamma(3, size=200)
    >>> fit = LogConcaveDensity().fit(x)
    >>> len(fit.knots_), np.exp(fit.score_samples([1, 3, 6])).round(3).tolist()
    (7, [0.186, 0.22, 0.039])
    >>> fit.cdf([1, 3, 6]).round(3).tolist()
    [0.068, 0.612, 0.943]
    """

    def fit(self, X, y=None):
        """Fit to the one-dimensional sample ``X``; ``y`` is ignored. Returns ``self``."""
        x, counts = np.unique(as_1d(X), return_counts=True)
        if len(x) < 2:
            raise ValueError("needs at least two distinct observations")
        w = counts / counts.sum()
        # Scaling by a power of two avoids rounding distinct floats together
        # when their ratios are representable.
        _, exponent = np.frexp(np.max(np.abs(x)))
        scale = np.ldexp(1.0, exponent - 1)
        support = x / scale
        if np.any(np.diff(support) == 0):
            raise ValueError("the sample spans too many scales to distinguish its observations")
        phi, knots = _active_set(support, w)
        self._unit_scale, self._unit_support, self._unit_phi = scale, support, phi
        self.support_, self.weights_ = x, w
        self.log_density_ = phi - np.log(scale)
        self.knots_ = x[knots]
        return self

    def score_samples(self, X):
        """Log density at each value of ``X`` (``-inf`` outside the range of the sample)."""
        check_is_fitted(self)
        t = np.asarray(X, dtype=float).ravel()
        low, high = self.support_[[0, -1]]
        unit = np.clip(t, low, high) / self._unit_scale
        value = np.interp(unit, self._unit_support, self.log_density_)
        return np.where((t < low) | (t > high), -np.inf, value)

    def cdf(self, X):
        """Distribution function at each value of ``X``."""
        check_is_fitted(self)
        low, high = self.support_[[0, -1]]
        t = np.clip(np.asarray(X, dtype=float).ravel(), low, high) / self._unit_scale
        x, phi = self._unit_support, self._unit_phi
        j = np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
        dx = np.diff(x)
        start = np.r_[0, np.cumsum(dx * _j00(phi[:-1], phi[1:]))]
        length = np.clip(t - x[j], 0, dx[j])
        change = (phi[j + 1] - phi[j]) * (length / dx[j])
        return np.clip(start[j] + np.exp(phi[j]) * length * _expm1_ratio(change), 0, 1)


class ScoreMatchingRegressor(RegressorMixin, BaseEstimator):
    """Linear regression by antitonic score matching (Feng, Kao, Xu & Samworth 2024).

    Among M-estimators with a convex loss, the smallest asymptotic variance
    is reached by the loss whose derivative is minus the antitonic projected
    score ``psi*``: the decreasing function closest to the score
    ``(log p)'`` of the error density ``p`` in mean square. ``psi* = (log
    p)'`` when ``p`` is log-concave, and ``psi*`` is bounded when ``p`` has
    heavier tails than an exponential distribution, so the estimate resists
    outliers in the response (not leverage points). The estimate needs no
    tuning: from the residuals of a least absolute deviations fit, ``psi*``
    is estimated by the slopes of the least concave majorant of ``J = p o
    F^-1``, with ``p`` and ``F`` from a normal kernel density estimate
    (bandwidth ``bw.nrd0``), on a grid of ``max(3000, 2n)`` quantiles. The
    coefficients minimize the convex loss by Newton steps, and the score is
    re-estimated once from the new residuals, as R's ``asm``.

    Parameters
    ----------
    symmetric : bool, default True
        Assume errors symmetric about zero: the score is symmetrized and the
        intercept is estimated with the slopes. Otherwise the slopes are
        estimated with centred predictors and the intercept is the median of
        ``y - X @ coef_``, with the variance of a sample median.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    scale_ : float
        MAD of the residuals, which standardizes them in the diagnostic plots.
    information_ : float
        Mean of ``psi(residuals)^2``, the estimated Fisher information
        ``int psi*^2 p`` of the projected score.
    cov_params_ : ndarray of shape (n_features + 1, n_features + 1)
        Asymptotic covariance of ``[intercept_, *coef_]``: ``(information_ X'X)^-1``
        for symmetric errors.
    std_errors_, z_values_, p_values_ : ndarray of shape (n_features + 1,)
        Standard errors, z statistics and two-sided normal p-values.
    score_knots_, score_values_ : ndarray
        The estimated score before symmetrization: linear between these
        points and constant beyond them. :meth:`psi` evaluates it.

    Notes
    -----
    The density of the residuals is binned on ``2^15`` points, as in ``asm``,
    and the fits agree with ``asm`` to about ``1e-6``. When more than half the
    residuals are equal, ``asm`` fails and the density is estimated over the
    range of all residuals. The variance with ``symmetric=False`` follows
    ``asm`` with its ``"median"`` intercept; by default ``asm`` uses the mean
    of the residuals.

    The efficiency result in the reference is for a sample-split estimator
    under its regularity conditions. This implementation uses the two-pass
    fit on the whole sample. For an exact fit, the score and inference
    are unavailable and reported as NaN.

    References
    ----------
    Feng, O. Y., Kao, Y.-C., Xu, M. and Samworth, R. J. (2024). Optimal
    convex M-estimation via score matching. arXiv:2403.16688.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.shape import ScoreMatchingRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2))
    >>> y = 1 + X @ [2.0, -1.0] + rng.standard_cauchy(300)  # least squares: (-3.09, -1.23, -6.55)
    >>> fit = ScoreMatchingRegressor().fit(X, y)
    >>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
    (1.01, [1.92, -0.9])
    >>> fit.std_errors_.round(3).tolist()
    [0.105, 0.104, 0.106]
    """

    fit_intercept = True  # read by the coefficient table

    def __init__(self, *, symmetric=True):
        self.symmetric = symmetric

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True, ensure_min_samples=3)
        n, p = X.shape
        if n <= p + 1:
            raise ValueError(f"needs more observations than coefficients (got n={n}, p={p + 1})")
        X, back = standardize_columns(X)  # well conditioned; the fit is affine equivariant
        mean = X.mean(axis=0)
        if np.linalg.matrix_rank(X - mean) < p:
            raise ValueError("X is singular: its columns (and the intercept) are collinear")
        design = np.column_stack([np.ones(n), X])
        beta = _l1_fit(design, y)  # least absolute deviations pilot
        for _ in range(2):  # estimate the score, fit, and repeat once from the new residuals
            fitted = y - design @ beta
            if is_exact_fit(y, fitted):
                score = None
                break
            score = _AntitonicScore(fitted, self.symmetric)
            if self.symmetric:
                beta = _minimize_loss(design, y, beta, score)
                fitted = y - design @ beta
            else:
                offset = beta[0] + mean @ beta[1:]
                slopes = _minimize_loss(X - mean, y - offset, beta[1:], score)
                fitted = y - offset - (X - mean) @ slopes  # the residuals the score was fitted to
                beta = np.r_[np.median(y - X @ slopes), slopes]
        self.scale_ = mad(y - design @ beta)
        self.information_ = 0.0 if score is None else float(np.mean(score(fitted)[0] ** 2))
        self.score_knots_ = np.empty(0) if score is None else score.knots
        self.score_values_ = np.empty(0) if score is None else score.values
        if self.information_ == 0:
            cov = np.full((p + 1, p + 1), np.nan)
        elif self.symmetric:
            cov = linalg.pinvh(self.information_ * design.T @ design)
        else:
            slopes = linalg.pinvh(self.information_ * np.cov(X, rowvar=False).reshape(p, p)) / n
            tau = 1 / (4 * score.median_density**2)  # n var(median)
            cov = np.empty((p + 1, p + 1))
            cov[0, 0], cov[1:, 1:] = tau / n + mean @ slopes @ mean, slopes
            cov[0, 1:] = cov[1:, 0] = -slopes @ mean
        beta, cov = back @ beta, back @ cov @ back.T
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        self.cov_params_ = cov
        self.std_errors_ = np.sqrt(np.diag(cov))
        self.z_values_ = beta / self.std_errors_
        self.p_values_ = 2 * stats.norm.sf(np.abs(self.z_values_))
        return self

    def predict(self, X):
        """Predicted values ``X @ coef_ + intercept_``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_

    def psi(self, residuals):
        """Estimated antitonic score at ``residuals``, or NaN for an exact fit."""
        check_is_fitted(self)
        if not len(self.score_knots_):
            return np.full_like(residuals, np.nan, dtype=float)
        score = _AntitonicScore.from_knots(self.score_knots_, self.score_values_, self.symmetric)
        return score(np.asarray(residuals, dtype=float))[0]

    def summary(self, feature_names=None):
        """Coefficient table, as R's ``summary(asm)``.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names of the columns of ``X``; ``x0, x1, ...`` by default.

        Returns
        -------
        str
        """
        check_is_fitted(self)
        return _coefficient_table(self, feature_names, "z", self.z_values_)


def _expm1_ratio(u):
    """``(exp(u) - 1) / u``, with the limit 1 at zero."""
    small = np.abs(u) < 1e-8
    return np.where(small, 1 + u / 2, np.expm1(u) / np.where(small, 1.0, u))


def _j00(a, b):
    """``int_0^1 exp((1 - t) a + t b) dt``."""
    return np.exp(a) * _expm1_ratio(b - a)


def _series(a, b, threshold, exact, coefficients):
    """``exp(a) g(b - a)`` with ``g`` exact, or by its Taylor series for small ``b - a``."""
    d = b - a
    small = np.abs(d) <= threshold
    safe = np.where(small, 1.0, d)
    series = sum(c * d**k for k, c in enumerate(coefficients))
    return np.exp(a) * np.where(small, series, exact(safe))


def _j10(a, b):
    """``int_0^1 (1 - t) exp((1 - t) a + t b) dt``."""
    return _series(
        a, b, 0.01, lambda d: (np.expm1(d) - d) / d**2, [1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720]
    )


def _j11(a, b):
    """``int_0^1 t (1 - t) exp((1 - t) a + t b) dt``."""
    return _series(
        a, b, 0.02, lambda d: (d * (np.exp(d) + 1) - 2 * np.expm1(d)) / d**3,
        [1 / 6, 1 / 12, 1 / 40, 1 / 180, 1 / 1008],
    )  # fmt: skip


def _j20(a, b):
    """``int_0^1 (1 - t)^2 exp((1 - t) a + t b) dt``."""
    return _series(
        a, b, 0.02, lambda d: 2 * (np.expm1(d) - d - d**2 / 2) / d**3,
        [1 / 3, 1 / 12, 1 / 60, 1 / 360, 1 / 2520],
    )  # fmt: skip


def _interpolate(x, knots, values):
    """Piecewise linear function through ``(x[knots], values)`` at ``x``."""
    return np.interp(x, x[knots], values)


def _knot_mle(x, w, knots, start):
    """Values at the knots of the log-density, linear between them, of largest likelihood.

    Maximizes ``sum_i w_i phi(x_i) - int exp(phi)`` by Newton steps with step
    halving; the second term is ``sum_k dx_k J00(phi_k, phi_k+1)`` and the
    Hessian is tridiagonal. The maximizer integrates to one.
    """
    xk = x[knots]
    left = np.clip(np.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    share = (x - xk[left]) / (xk[left + 1] - xk[left])
    wk = np.bincount(left, w * (1 - share), len(xk)) + np.bincount(left + 1, w * share, len(xk))
    dx = np.diff(xk)

    def loglik(phi):
        return wk @ phi - dx @ _j00(phi[:-1], phi[1:])

    phi = start - np.log(dx @ _j00(start[:-1], start[1:]))  # normalized start
    for _ in range(200):  # pragma: no branch (Newton steps converge long before the cap)
        a, b = phi[:-1], phi[1:]
        grad = wk.copy()
        grad[:-1] -= dx * _j10(a, b)
        grad[1:] -= dx * _j10(b, a)
        diagonal = np.zeros_like(phi)
        diagonal[:-1] += dx * _j20(a, b)
        diagonal[1:] += dx * _j20(b, a)
        off = dx * _j11(a, b)
        step = linalg.solveh_banded(np.vstack([np.r_[0, off], diagonal]), grad)
        decrement = grad @ step
        if decrement <= 1e-14:
            break
        t, base = 1.0, loglik(phi)
        while loglik(phi + t * step) < base + t * decrement / 4 and t > 1e-10:
            t /= 2
        phi = phi + t * step
    return phi


def _kinks(x, phi):
    """Change of slope of ``phi`` at each point (zero at the extremes)."""
    return np.r_[0, np.diff(np.diff(phi) / np.diff(x)), 0]


def _directional_derivatives(x, w, phi, knots):
    """Derivative of the log-likelihood towards a concave kink at each observation.

    The direction is the tent that vanishes at the neighbouring knots and has
    height ``D lambda (1 - lambda)`` at the observation, ``lambda`` its
    relative position in the gap ``D`` between knots (Dümbgen, Hüsler &
    Rufibach 2007); knots get zero.
    """
    out = np.zeros(len(x))
    k = np.flatnonzero(knots)
    for low, high in zip(k[:-1], k[1:], strict=True):
        if high > low + 1:
            gap = x[high] - x[low]
            inner = slice(low + 1, high)
            lam, wi = (x[inner] - x[low]) / gap, w[inner]
            data = np.cumsum(wi * lam) - lam * np.cumsum(wi) + lam * np.sum(wi * (1 - lam))
            mass = lam * _j10(phi[inner], phi[low]) + (1 - lam) * _j10(phi[inner], phi[high])
            out[inner] = gap * data - gap**2 * lam * (1 - lam) * mass
    return out


def _active_set(x, w, precision=1e-10):
    """Log-concave MLE ``phi`` at ``x`` and its knots, as R's ``logcondens::activeSetLogCon``."""
    knots = np.zeros(len(x), dtype=bool)
    knots[[0, -1]] = True
    phi = _interpolate(x, knots, _knot_mle(x, w, knots, np.zeros(2)))
    for _ in range(max(500, len(x))):  # pragma: no branch (each pass raises the likelihood)
        gains = _directional_derivatives(x, w, phi, knots)
        if gains.max() <= precision * np.mean(np.abs(gains)):
            break
        knots[np.argmax(gains)] = True
        new = _interpolate(x, knots, _knot_mle(x, w, knots, phi[knots]))
        kinks = _kinks(x, new) * knots
        while kinks.max() > precision * np.abs(kinks).max():  # not concave: step back
            convex = kinks > 0
            current = np.minimum(_kinks(x, phi)[convex], 0)
            ratio = current / (current - kinks[convex])
            lam = ratio.min()
            knots[np.flatnonzero(convex)[ratio == lam]] = False
            phi = phi + lam * (new - phi)
            new = _interpolate(x, knots, _knot_mle(x, w, knots, phi[knots]))
            kinks = _kinks(x, new) * knots
        phi = new
    return phi, knots


def _nrd0(x):
    """Silverman's rule of thumb, R's ``bw.nrd0``."""
    q1, q3 = np.quantile(x, [0.25, 0.75])
    spread = min(np.std(x, ddof=1), (q3 - q1) / 1.34)
    if spread == 0:
        spread = np.std(x, ddof=1) or abs(x[0]) or 1.0
    return 0.9 * spread * len(x) ** -0.2


def _binned_density(x, h, grid):
    """Normal kernel density estimate on an even ``grid``, by linear binning and FFT (Wand 1994).

    Observations beyond the grid are dropped, as R's ``density``.
    """
    m, step = len(grid), grid[1] - grid[0]
    position = (x - grid[0]) / step
    position = position[(position >= 0) & (position <= m - 1)]
    cell = np.minimum(position.astype(int), m - 2)
    share = position - cell
    counts = np.bincount(cell, 1 - share, m) + np.bincount(cell + 1, share, m)
    kernel = stats.norm.pdf(np.arange(1 - m, m) * step, scale=h)
    return np.maximum(signal.fftconvolve(counts, kernel)[m - 1 : 2 * m - 1], 0) / len(x)


class _AntitonicScore:
    """Estimated antitonic projected score, piecewise linear and constant beyond its knots.

    ``J = p o F^-1`` on the quantile grid ``u_k = k / K``, ``K = max(3000,
    2n)``; the decreasing (isotonic) regression of its slopes gives the slopes
    of the least concave majorant of ``J``, which are the score at the
    midpoints of ``F^-1``. The cells of the ``2 floor(K / n)`` extreme
    quantiles at each end are dropped, as R's ``asm``. The density is
    estimated up to 20 interquartile ranges from the median (all residuals
    when more than half are equal, where ``asm`` fails).
    """

    def __init__(self, residuals, symmetric):
        r = np.sort(residuals)
        n = len(r)
        h = _nrd0(r)
        q1, median, q3 = np.quantile(r, [0.25, 0.5, 0.75])
        low, high = r[0] - 3 * h, r[-1] + 3 * h
        if q3 > q1:  # leave out residuals beyond 20 interquartile ranges from the median
            low, high = max(low, median - 20 * (q3 - q1)), min(high, median + 20 * (q3 - q1))
        grid = np.linspace(low, high, 2**15)
        density = _binned_density(r, h, grid)
        cdf = integrate.cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        last = np.r_[cdf[1:] > cdf[:-1], True]  # inverse takes the largest point of a flat part
        self.median_density = float(np.interp(np.interp(0.5, cdf[last], grid[last]), grid, density))
        k = max(3000, 2 * n)
        quantiles = np.interp(np.arange(k + 1) / k, cdf[last], grid[last])
        J = np.interp(quantiles, grid, density)
        values = isotonic_regression(np.diff(J) * k, increasing=False)
        knots = (quantiles[1:] + quantiles[:-1]) / 2
        skip = 2 * (k // n)
        if k - 2 * skip >= 2:
            knots, values = knots[skip : k - skip], values[skip : k - skip]
        jumps = np.diff(values) != 0
        keep = np.r_[jumps, True] | np.r_[True, jumps]
        self.knots, self.values, self.symmetric = knots[keep], values[keep], symmetric

    @classmethod
    def from_knots(cls, knots, values, symmetric):
        score = cls.__new__(cls)
        score.knots, score.values, score.symmetric = knots, values, symmetric
        return score

    def _one_sided(self, r):
        """Score, its slope and its antiderivative at ``r``."""
        knots, values = self.knots, self.values
        left, right = max(values[0], 1e-7), min(values[-1], -1e-7)
        psi = np.interp(r, knots, values, left=left, right=right)
        slopes = np.diff(values) / np.diff(knots)
        j = np.clip(np.searchsorted(knots, r, side="right") - 1, 0, len(knots) - 2)
        inside = (r >= knots[0]) & (r < knots[-1])
        slope = np.where(inside, slopes[j], 0.0)
        area = integrate.cumulative_trapezoid(values, knots, initial=0)
        clipped = np.clip(r, knots[0], knots[-1])
        within = (
            area[j] + (clipped - knots[j]) * (values[j] + np.interp(clipped, knots, values)) / 2
        )
        integral = (
            within + left * np.minimum(r - knots[0], 0) + right * np.maximum(r - knots[-1], 0)
        )
        return psi, slope, integral

    def __call__(self, r):
        """``(psi(r), psi'(r), loss(r))`` with ``loss' = -psi``, symmetrized if required."""
        psi, slope, integral = self._one_sided(r)
        if not self.symmetric:
            return psi, slope, -integral
        mirror_psi, mirror_slope, mirror_integral = self._one_sided(-r)
        return (psi - mirror_psi) / 2, (slope + mirror_slope) / 2, -(integral + mirror_integral) / 2


def _minimize_loss(X, y, beta, score):
    """Minimize ``sum loss(y - X b)`` (convex) by Newton steps with step halving."""
    for _ in range(200):  # pragma: no branch (Newton steps converge long before the cap)
        psi, slope, loss = score(y - X @ beta)
        grad = X.T @ psi  # gradient of the loss; its Hessian is X' diag(-psi') X
        step = np.linalg.lstsq(X.T @ (X * -slope[:, None]), grad, rcond=None)[0]
        decrement = grad @ step
        if decrement <= 0:  # flat Hessian along the gradient: take a gradient step
            step, decrement = grad / max(len(y), 1), grad @ grad / max(len(y), 1)
        t, base = 1.0, loss.sum()
        while score(y - X @ (beta - t * step))[2].sum() > base - t * decrement / 4 and t > 1e-12:
            t /= 2
        beta = beta - t * step
        if np.max(np.abs(t * step)) <= 1e-12 * (1 + np.max(np.abs(beta))):
            break
    return beta
