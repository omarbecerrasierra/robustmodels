"""Outlier detection by projection outlyingness and depth.

The detectors follow scikit-learn's outlier detection API, as
:class:`sklearn.covariance.EllipticEnvelope`. ``fit(X)`` takes observations in
rows and sets ``outlyingness_`` and ``cutoff_``; ``score_samples(X)`` returns the
projection depth ``1 / (1 + outlyingness)`` (higher is more central),
``decision_function(X)`` is negative for outliers and ``predict(X)`` returns -1
for outliers and 1 for inliers.
"""

import itertools
import math

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_generator, check_count
from .scale import medcouple

__all__ = ["AdjustedOutlyingness", "DirectionalOutlyingness", "StahelDonohoOutlyingness"]

_MAD = 1 / stats.norm.ppf(0.75)  # makes the MAD consistent at the normal model
_DO_TUNING = 2.1  # Rousseeuw, Raymaekers & Hubert (2018): 85% efficiency
# alpha: integral over (0, inf) of rho(t) = min((t / c)^2, 1) against the normal density
_DO_ALPHA = (
    stats.norm.cdf(_DO_TUNING) - 0.5 - _DO_TUNING * stats.norm.pdf(_DO_TUNING)
) / _DO_TUNING**2 + stats.norm.sf(_DO_TUNING)


class _ProjectionOutlyingness(OutlierMixin, BaseEstimator):
    """Base class: directions, depth, the cutoff and the scikit-learn methods."""

    def __init__(self, *, n_directions=None, random_state=None):
        self.n_directions = n_directions
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        n, p = X.shape
        if n <= p:
            raise ValueError(f"needs more observations than variables (got n={n}, p={p})")
        if self.n_directions is not None:
            check_count("n_directions", self.n_directions)
        n_directions = 250 * p if self.n_directions is None else self.n_directions
        rng = as_generator(self.random_state)
        self.directions_ = _directions(X, n_directions, rng)
        self._fit_projections(_merge_ties(X @ self.directions_))
        self.outlyingness_ = self.outlyingness(X)
        self.cutoff_ = _log_cutoff(self.outlyingness_)
        self.offset_ = 1 / (1 + self.cutoff_)
        return self

    def outlyingness(self, X):
        """Outlyingness of each row of ``X``: the largest standardized projection."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return self._outlyingness(X @ self.directions_).max(axis=1)

    def score_samples(self, X):
        """Projection depth in [0, 1]; higher is more central, 0 for infinite outlyingness."""
        return 1 / (1 + self.outlyingness(X))

    def decision_function(self, X):
        """``score_samples(X) - offset_``; negative for outliers."""
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """-1 for outliers (outlyingness above ``cutoff_``) and 1 for inliers."""
        return np.where(self.decision_function(X) < 0, -1, 1)


class StahelDonohoOutlyingness(_ProjectionOutlyingness):
    """Outlier detection by Stahel-Donoho outlyingness (Stahel 1981; Donoho 1982).

    The outlyingness of ``x`` is the largest ``|u'x - med(u'X)| / mad(u'X)``
    over directions ``u``, with the MAD consistent at the normal model; the
    projection depth is ``1 / (1 + outlyingness)`` (Zuo 2003). Directions are
    normal to hyperplanes through ``p`` observations, which makes the result
    affine invariant; with at most ``n_directions`` such hyperplanes all are
    used, otherwise a random selection, as R's mrfDepth. Observations with
    outlyingness above ``cutoff_`` are outliers. For skewed data use
    :class:`AdjustedOutlyingness`.

    Parameters
    ----------
    n_directions : int, default None
        Number of directions; ``250 * n_features`` by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions.

    Attributes
    ----------
    outlyingness_ : ndarray of shape (n_samples,)
        Outlyingness of the training data.
    cutoff_ : float
        ``exp(median(L) + mad(L) * z) - 0.1``, with ``L = log(0.1 + outlyingness_)``
        and ``z`` the 0.995 normal quantile, as mrfDepth.
    offset_ : float
        Depth at the cutoff, ``1 / (1 + cutoff_)``.
    directions_ : ndarray of shape (n_features, n_directions)
        Unit directions.
    centers_, scales_ : ndarray of shape (n_directions,)
        Median and MAD of the training projections on each direction.

    References
    ----------
    Stahel, W. A. (1981). *Robuste Schätzungen: infinitesimale Optimalität und
    Schätzungen von Kovarianzmatrizen*. PhD thesis, ETH Zürich.

    Donoho, D. L. (1982). *Breakdown properties of multivariate location
    estimators*. Qualifying paper, Harvard University.

    Zuo, Y. (2003). Projection-based depth functions and associated medians.
    *The Annals of Statistics*, 31(5), 1460-1490.

    Segaert, P., Hubert, M., Rousseeuw, P. J. and Raymaekers, J. (2018).
    mrfDepth: Depth measures in multivariate, regression and functional
    settings. R package.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.outliers import StahelDonohoOutlyingness
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:5] += 6  # five outliers
    >>> sdo = StahelDonohoOutlyingness(random_state=0).fit(X)
    >>> np.flatnonzero(sdo.predict(X) == -1).tolist()
    [0, 1, 2, 3, 4]
    """

    def _fit_projections(self, projections):
        self.centers_ = np.median(projections, axis=0)
        self.scales_ = _MAD * np.median(np.abs(projections - self.centers_), axis=0)

    def _outlyingness(self, projections):
        return _ratio(np.abs(projections - self.centers_), self.scales_)


class AdjustedOutlyingness(_ProjectionOutlyingness):
    """Outlier detection for skewed data by adjusted outlyingness (Hubert & Van der Veeken 2008).

    On each direction ``u``, the distance of ``u'x`` to the median of the
    projections is divided by the distance from the median to the whisker of
    the skewness-adjusted boxplot on the same side (Hubert & Vandervieren
    2008), which uses the medcouple ``MC``: the fences are
    ``Q3 + 1.5 exp(3 MC) IQR`` and ``Q1 - 1.5 exp(-4 MC) IQR`` for ``MC >= 0``
    (``exp(4 MC)`` and ``exp(-3 MC)`` otherwise). The outlyingness is the
    largest ratio over the directions, and ``1 / (1 + outlyingness)`` is the
    skew-adjusted projection depth. Because the scale on each side follows the
    skewness, outliers on the short side of a skewed distribution are not masked
    by the long tail, as they are for :class:`StahelDonohoOutlyingness`. The
    medcouple follows its definition, as robustbase; mrfDepth's differs when
    projections are tied, which affine directions always produce. Quartiles use the
    ``(k - 0.5) / n`` rule (R's type 5), as mrfDepth; directions and cutoff as
    in :class:`StahelDonohoOutlyingness`.

    Parameters
    ----------
    n_directions : int, default None
        Number of directions; ``250 * n_features`` by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions.

    Attributes
    ----------
    outlyingness_ : ndarray of shape (n_samples,)
        Adjusted outlyingness of the training data.
    cutoff_ : float
        ``exp(median(L) + mad(L) * z) - 0.1``, with ``L = log(0.1 + outlyingness_)``
        and ``z`` the 0.995 normal quantile, as mrfDepth.
    offset_ : float
        Depth at the cutoff, ``1 / (1 + cutoff_)``.
    directions_ : ndarray of shape (n_features, n_directions)
        Unit directions.
    centers_ : ndarray of shape (n_directions,)
        Median of the training projections on each direction.
    lower_whiskers_, upper_whiskers_ : ndarray of shape (n_directions,)
        Distances from the median to the adjusted boxplot whiskers.

    References
    ----------
    Hubert, M. and Van der Veeken, S. (2008). Outlier detection for skewed data.
    *Journal of Chemometrics*, 22(3-4), 235-246.

    Hubert, M. and Vandervieren, E. (2008). An adjusted boxplot for skewed
    distributions. *Computational Statistics & Data Analysis*, 52(12),
    5186-5201.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.outliers import AdjustedOutlyingness
    >>> X = np.random.default_rng(0).lognormal(size=(100, 2))
    >>> X[:5] = [-3, -3]  # five outliers on the short side
    >>> ao = AdjustedOutlyingness(random_state=0).fit(X)
    >>> np.flatnonzero(ao.predict(X) == -1).tolist()
    [0, 1, 2, 3, 4]
    """

    def _fit_projections(self, projections):
        self.centers_ = np.median(projections, axis=0)
        whiskers = [_whiskers(column) for column in (projections - self.centers_).T]
        self.lower_whiskers_, self.upper_whiskers_ = np.array(whiskers).T

    def _outlyingness(self, projections):
        y = projections - self.centers_
        whisker = np.where(y > 0, self.upper_whiskers_, self.lower_whiskers_)
        return _ratio(np.abs(y), whisker)


class DirectionalOutlyingness(_ProjectionOutlyingness):
    """Outlier detection for skewed data by directional outlyingness (Rousseeuw et al. 2018).

    On each direction ``u``, the distance of ``u'x`` to the median of the
    projections is divided by a scale of the half sample on the same side:
    ``s_a`` above the median and ``s_b`` below. Each is a one-step M-estimator
    of scale of the deviations from the median, ``s = s_o sqrt(sum(rho(z_i /
    s_o)) / (2 alpha h))``, with ``s_o`` the median of the deviations divided
    by the 0.75 normal quantile, ``rho(t) = min((t / 2.1)^2, 1)`` and ``h =
    (n + 1) // 2`` deviations per side; ``alpha`` makes both scales consistent
    at the normal model. The outlyingness is the largest ratio over the
    directions. Unlike :class:`AdjustedOutlyingness` it needs no medcouple, so
    it is faster and its scales cannot be inflated by a few far points.
    R's mrfDepth uses ``1.54`` for ``1 / (2 sqrt(alpha)) = 1.534`` and leaves
    the median out of the half samples when ``n`` is odd; this class follows
    the paper. Directions and cutoff as in :class:`StahelDonohoOutlyingness`.

    Parameters
    ----------
    n_directions : int, default None
        Number of directions; ``250 * n_features`` by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions.

    Attributes
    ----------
    outlyingness_ : ndarray of shape (n_samples,)
        Directional outlyingness of the training data.
    cutoff_ : float
        ``exp(median(L) + mad(L) * z) - 0.1``, with ``L = log(0.1 + outlyingness_)``
        and ``z`` the 0.995 normal quantile.
    offset_ : float
        Depth at the cutoff, ``1 / (1 + cutoff_)``.
    directions_ : ndarray of shape (n_features, n_directions)
        Unit directions.
    centers_ : ndarray of shape (n_directions,)
        Median of the training projections on each direction.
    lower_scales_, upper_scales_ : ndarray of shape (n_directions,)
        Scales below and above the median.

    References
    ----------
    Rousseeuw, P. J., Raymaekers, J. and Hubert, M. (2018). A measure of
    directional outlyingness with applications to image data and video.
    *Journal of Computational and Graphical Statistics*, 27(2), 345-359.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.outliers import DirectionalOutlyingness
    >>> X = np.random.default_rng(0).lognormal(size=(100, 2))
    >>> X[:5] = [-3, -3]  # five outliers on the short side
    >>> do = DirectionalOutlyingness(random_state=0).fit(X)
    >>> np.flatnonzero(do.predict(X) == -1).tolist()
    [0, 1, 2, 3, 4]
    """

    def _fit_projections(self, projections):
        ordered = np.sort(projections, axis=0)
        half = (len(ordered) + 1) // 2
        self.centers_ = np.median(ordered, axis=0)
        self.lower_scales_ = _half_scale(self.centers_ - ordered[:half])
        self.upper_scales_ = _half_scale(ordered[-half:] - self.centers_)

    def _outlyingness(self, projections):
        y = projections - self.centers_
        return _ratio(np.abs(y), np.where(y > 0, self.upper_scales_, self.lower_scales_))


def _half_scale(deviations):
    """One-step M-scale of each column of nonnegative deviations from the median."""
    initial = _MAD * np.median(deviations, axis=0)
    rho = np.minimum((_ratio(deviations, initial) / _DO_TUNING) ** 2, 1)
    return initial * np.sqrt(rho.mean(axis=0) / (2 * _DO_ALPHA))


def _directions(X, n_directions, rng):
    """Unit normals of hyperplanes through ``p`` observations, as columns.

    All ``p``-subsets when there are at most ``n_directions``, otherwise random
    ones; subsets that do not span a hyperplane are skipped.
    """
    n, p = X.shape
    if p == 1:
        return np.ones((1, 1))
    if math.comb(n, p) <= n_directions:
        subsets = itertools.combinations(range(n), p)
    else:
        subsets = (rng.choice(n, p, replace=False) for _ in range(100 * n_directions))
    normals = []
    for subset in subsets:
        normal = _normal(X[list(subset)])
        if normal is not None:
            normals.append(normal)
            if len(normals) == n_directions:
                break
    if not normals:
        raise ValueError("X is singular: the observations lie on a hyperplane")
    return np.array(normals).T


def _normal(points):
    """Unit normal of the hyperplane through ``p`` points in ``p`` dimensions, or None."""
    _, singular, vt = np.linalg.svd(points[1:] - points[0])
    if singular[-1] <= 1e-12 * singular[0]:
        return None
    return vt[-1]


def _merge_ties(projections, tol=1e-10):
    """Make projections that differ only by rounding exactly equal, column by column.

    The observations that define a direction, and any others on the same
    hyperplane, project to one value; rounding must not break those ties, which
    the medcouple treats specially. Values within ``tol`` times the range of
    their column of the previous one form a group and take its middle value.
    """
    n, k = projections.shape
    index = np.arange(n)[:, None]
    order = np.argsort(projections, axis=0, kind="stable")
    ordered = np.take_along_axis(projections, order, axis=0)
    span = ordered[-1] - ordered[0]
    starts = np.vstack([np.ones((1, k), bool), np.diff(ordered, axis=0) > tol * span])
    ends = np.vstack([starts[1:], np.ones((1, k), bool)])
    first = np.maximum.accumulate(np.where(starts, index, 0), axis=0)
    last = np.minimum.accumulate(np.where(ends, index, n - 1)[::-1], axis=0)[::-1]
    middle = np.take_along_axis(ordered, (first + last) // 2, axis=0)
    merged = np.empty_like(projections)
    np.put_along_axis(merged, order, middle, axis=0)
    return merged


def _ratio(distance, scale):
    """``distance / scale``: 0 where the distance is 0, infinite where only the scale is."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(distance == 0, 0.0, distance / scale)


def _whiskers(y):
    """Distances from the median (0) to the adjusted boxplot whiskers of ``y``."""
    q1, q3 = np.percentile(y, [25, 75], method="hazen")
    iqr = q3 - q1
    mc = medcouple(y)
    upper = q3 + 1.5 * iqr * np.exp(3 * mc if mc >= 0 else 4 * mc)
    lower = q1 - 1.5 * iqr * np.exp(-4 * mc if mc >= 0 else -3 * mc)
    return -y[y >= lower].min(), y[y <= upper].max()


def _log_cutoff(outlyingness):
    """Cutoff of mrfDepth: a robust 99.5% bound for ``log(0.1 + outlyingness)``."""
    logs = np.log(0.1 + outlyingness)
    center = np.median(logs)
    spread = _MAD * np.median(np.abs(logs - center))
    return float(np.exp(center + spread * stats.norm.ppf(0.995)) - 0.1)
