"""Robust covariance estimators and their fitting steps.

The estimators follow the scikit-learn API and extend
:class:`sklearn.covariance.EmpiricalCovariance`. ``fit(X)`` takes observations
in rows and sets ``location_``, ``covariance_``, ``precision_`` and ``dist_``
(squared robust distances of the training data). The inherited
``mahalanobis(X)`` returns squared robust distances of new observations, and
``score``, ``error_norm`` and ``get_precision`` work as in scikit-learn.
"""

import math
import warnings

import numpy as np
from scipy import linalg, stats
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import validate_data

from ._validation import as_generator, check_count, check_option, check_range
from .outliers import StahelDonohoOutlyingness
from .scale import (
    _bisquare_weights,
    _mscale_bisquare,
    _rho,
    _tau_estimates,
    _tuning_for_breakdown,
    _tuning_for_shape_efficiency,
    mad,
    pn,
    qn,
    tau_scale,
)

__all__ = [
    "MCD",
    "MRCD",
    "MVE",
    "OGK",
    "MMEstimator",
    "SEstimator",
    "StahelDonoho",
    "h_alpha_n",
    "mcd_consistency",
    "mcd_small_sample",
]


class _RobustCovariance(EmpiricalCovariance):
    """Base class; ``mahalanobis``, ``score`` and ``error_norm`` come from scikit-learn."""

    store_precision = True  # read by the inherited methods
    assume_centered = False

    def _validate(self, X):
        return validate_data(self, X, dtype=np.float64, ensure_min_samples=2)

    def _set_estimates(self, location, covariance, X):
        self.location_ = np.asarray(location, dtype=float)
        self.covariance_ = np.asarray(covariance, dtype=float)
        self.precision_ = linalg.pinvh(self.covariance_)
        diff = X - self.location_
        self.dist_ = np.sum((diff @ self.precision_) * diff, axis=1)


class MCD(_RobustCovariance):
    """Minimum Covariance Determinant estimator (Rousseeuw 1984).

    Mean and covariance of the ``h`` observations whose covariance matrix has the
    smallest determinant, scaled for consistency at the normal model. The
    reweighting step then recomputes both from the observations whose squared
    robust distance is below the 0.975 chi-square quantile. In general position,
    the exact raw estimator has breakdown ``min(n - h + 1, h - p) / n``,
    approaching 50% for ``alpha = 0.5`` and fixed ``p``. The multivariate
    search does not certify a global minimum. Needs ``n > p``; use
    :class:`MRCD` otherwise. If ``h`` observations lie on a hyperplane, a
    ``RuntimeWarning`` reports the exact fit and the raw estimates are kept.

    Parameters
    ----------
    alpha : float in [0.5, 1], default 0.5
        Coverage. The subset size ``h = h_alpha_n(alpha, n, p)`` goes from
        ``(n + p + 1) // 2`` (maximal breakdown) at 0.5 to ``n`` at 1.
    algorithm : {"fast", "deterministic"}, default "fast"
        FAST-MCD with random starts (Rousseeuw & Van Driessen 1999), or DetMCD
        with six deterministic starts (Hubert, Rousseeuw & Verdonck 2012).
        With one variable both use the exact univariate MCD.
    n_trials : int, default 500
        Random starting subsets of FAST-MCD.
    reweighting : bool, default True
        Apply the reweighting step.
    small_sample_correction : bool, default True
        Multiply the covariance by the factors of Pison, Van Aelst & Willems (2002).
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of FAST-MCD.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Final location.
    covariance_ : ndarray of shape (n_features, n_features)
        Final covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Pseudo-inverse of ``covariance_``.
    support_ : ndarray of bool, shape (n_samples,)
        Observations used for the final estimates.
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    raw_location_ : ndarray of shape (n_features,)
        Mean of the ``h``-subset.
    raw_covariance_ : ndarray of shape (n_features, n_features)
        Corrected covariance of the ``h``-subset.
    raw_support_ : ndarray of bool, shape (n_samples,)
        The ``h``-subset.
    h_ : int
        Subset size.

    References
    ----------
    Rousseeuw, P. J. (1984). Least median of squares regression. *Journal of the
    American Statistical Association*, 79(388), 871-880.

    Rousseeuw, P. J. and Van Driessen, K. (1999). A fast algorithm for the
    minimum covariance determinant estimator. *Technometrics*, 41(3), 212-223.

    Hubert, M., Rousseeuw, P. J. and Verdonck, T. (2012). A deterministic
    algorithm for robust location and scatter. *Journal of Computational and
    Graphical Statistics*, 21(3), 618-637.

    Croux, C. and Haesbroeck, G. (1999). Influence function and efficiency of the
    minimum covariance determinant scatter matrix estimator. *Journal of
    Multivariate Analysis*, 71(2), 161-190.

    Pison, G., Van Aelst, S. and Willems, G. (2002). Small sample corrections for
    LTS and MCD. *Metrika*, 55, 111-123.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import MCD
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:10] += 10  # ten outliers
    >>> mcd = MCD(random_state=0).fit(X)
    >>> mcd.location_.round(2).tolist()
    [0.01, 0.1]
    >>> int(mcd.support_[:10].sum())  # no outlier is used
    0
    """

    def __init__(
        self,
        *,
        alpha=0.5,
        algorithm="fast",
        n_trials=500,
        reweighting=True,
        small_sample_correction=True,
        random_state=None,
    ):
        self.alpha = alpha
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.reweighting = reweighting
        self.small_sample_correction = small_sample_correction
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        n, p = X.shape
        check_option("algorithm", self.algorithm, ("fast", "deterministic"))
        check_count("n_trials", self.n_trials)
        if n <= p:
            raise ValueError(f"MCD needs n > p (got n={n}, p={p}); use MRCD")
        h = h_alpha_n(self.alpha, n, p)
        subset = self._best_subset(X, h)
        self._set_raw_estimates(X, subset, h)
        location, covariance = self._reweighted_estimates(X)
        self._set_estimates(location, covariance, X)
        return self

    def _set_raw_estimates(self, X, subset, h):
        n, p = X.shape
        location, covariance = _mean_cov(X[subset])
        if h < n:
            covariance *= mcd_consistency(p, h / n) * self._small_sample(p, n, False)
        self.h_ = h
        self.raw_location_ = location
        self.raw_covariance_ = covariance
        self.raw_support_ = np.isin(np.arange(n), subset)

    def _reweighted_estimates(self, X):
        location, covariance = self.raw_location_, self.raw_covariance_
        self.support_ = self.raw_support_.copy()
        if _logdet(covariance) == -np.inf:
            warnings.warn(
                "exact fit: h observations lie on a hyperplane", RuntimeWarning, stacklevel=3
            )
            return location, covariance
        if self.reweighting:
            location, covariance, self.support_ = _reweight(X, location, covariance)
            if not self.support_.all():
                n, p = X.shape
                covariance = covariance * self._small_sample(p, n, True)
        return location, covariance

    def _best_subset(self, X, h):
        n, p = X.shape
        if h == n:
            return np.arange(n)
        center, scale = _center_scale(X)
        standardized = (X - center) / scale
        if p == 1:
            return _univariate_mcd(standardized[:, 0], h)
        if self.algorithm == "fast":
            rng = as_generator(self.random_state)
            return _fast_mcd(standardized, h, self.n_trials, rng)
        return _best_mcd(standardized, _deterministic_starts(standardized, h), h)

    def _small_sample(self, p, n, reweighted):
        if not self.small_sample_correction:
            return 1.0
        return mcd_small_sample(p, n, self.alpha, reweighted)


def _mcd_steps(X, subset, h, max_steps=200):
    """Concentrate a subset while its covariance determinant decreases."""
    location, covariance = _mean_cov(X[subset])
    determinant = _logdet(covariance)
    for _ in range(max_steps):
        if determinant == -np.inf:
            break
        distances = _mahalanobis_sq(X, location, covariance)
        updated = np.sort(np.argpartition(distances, h - 1)[:h])
        new_location, new_covariance = _mean_cov(X[updated])
        new_determinant = _logdet(new_covariance)
        if len(subset) == h and new_determinant >= determinant:
            break
        subset = updated
        location, covariance = new_location, new_covariance
        determinant = new_determinant
    return determinant, subset


def _best_mcd(X, starts, h, max_steps=200):
    """Return the best subset after concentrating each starting subset."""
    candidates = [_mcd_steps(X, subset, h, max_steps) for subset in starts]
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _fast_mcd(X, h, n_trials, rng):
    """Take two C-steps per random start, then converge the ten best subsets."""
    starts = [_random_subset(X, rng) for _ in range(n_trials)]
    if starts[0] is None:
        return np.arange(h)
    candidates = [_mcd_steps(X, subset, h, max_steps=2) for subset in starts]
    candidates.sort(key=lambda candidate: candidate[0])
    finalists = [subset for _, subset in candidates[:10]]
    return _best_mcd(X, finalists, h)


def _deterministic_starts(Z, h):
    """The six h-subsets of DetMCD, from robustly standardized data ``Z``."""
    n, p = Z.shape
    ranks = stats.rankdata(Z, axis=0)
    norms = np.linalg.norm(Z, axis=1)
    signs = Z / np.where(norms > 0, norms, 1)[:, None]
    scatters = [
        _corr(np.tanh(Z)),
        _corr(ranks),  # Spearman
        _corr(stats.norm.ppf((ranks - 1 / 3) / (n + 1 / 3))),  # normal scores
        signs.T @ signs,  # spatial sign covariance
        _mean_cov(Z[np.argsort(norms)[: max(2, math.ceil(n / 2))]])[1],  # first BACON step
        _gk_covariance(Z, qn),  # OGK-type
    ]
    return [_start_from_scatter(Z, S, h) for S in scatters]


def _corr(Y):
    # A constant column contributes zero correlation to a starting scatter.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.atleast_2d(np.corrcoef(Y, rowvar=False))
    return np.nan_to_num(correlation)


def _start_from_scatter(Z, S, h):
    """h points closest to a robust center, in the eigenbasis of ``S`` with Qn scales."""
    eigenvectors = np.linalg.eigh(S)[1]
    projected = Z @ eigenvectors
    scales = np.array([qn(column) for column in projected.T])
    scales = np.where(scales > 0, scales, 1e-300)
    whitening = (eigenvectors / scales) @ eigenvectors.T
    inverse_whitening = (eigenvectors * scales) @ eigenvectors.T
    center = np.median(Z @ whitening, axis=0) @ inverse_whitening
    residuals = ((Z - center) @ eigenvectors) / scales
    distances = np.sum(residuals**2, axis=1)
    return np.sort(np.argsort(distances, kind="stable")[:h])


def _univariate_mcd(z, h):
    """Exact MCD for p = 1: the window of h sorted values with the least variance."""
    order = np.argsort(z, kind="stable")
    sorted_values = z[order]
    # Two stacks form a sliding queue of Welford summaries. Unlike subtracting
    # prefix sums of squares, this never subtracts an extreme departed outlier
    # from a small window variance. Each observation is pushed/popped O(1) times.
    front, back = [], []

    def push(stack, value):
        count, mean, m2 = stack[-1][1:] if stack else (0, 0.0, 0.0)
        delta = value - mean
        updated_mean = mean + delta / (count + 1)
        updated_m2 = m2 + delta * (value - updated_mean)
        stack.append((value, count + 1, updated_mean, updated_m2))

    best, start = np.inf, 0
    for end, value in enumerate(sorted_values):
        push(back, value)
        if end >= h:
            if not front:
                while back:
                    push(front, back.pop()[0])
            front.pop()
        if end < h - 1:
            continue
        if not front:
            m2 = back[-1][3]
        elif not back:
            m2 = front[-1][3]
        else:
            _, n1, mean1, m21 = front[-1]
            _, n2, mean2, m22 = back[-1]
            m2 = m21 + m22 + (mean1 - mean2) ** 2 * n1 * n2 / h
        if m2 < best:
            best, start = m2, end - h + 1
    return np.sort(order[start : start + h])


def h_alpha_n(alpha, n, p):
    """Subset size of the MCD for coverage ``alpha``, as R's ``robustbase::h.alpha.n``.

    ``(n + p + 1) // 2`` at ``alpha = 0.5``, ``n`` at ``alpha = 1`` and linear in
    between.

    Parameters
    ----------
    alpha : float in [0.5, 1]
        Coverage.
    n, p : int
        Number of observations and of variables.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``n <= p``: no subset of at most ``n`` observations exists.

    Examples
    --------
    >>> from robustmodels.covariance import h_alpha_n
    >>> h_alpha_n(0.5, 100, 3), h_alpha_n(0.75, 100, 3)
    (52, 76)
    """
    check_range("alpha", alpha, 0.5, 1)
    check_count("n", n)
    check_count("p", p)
    if n <= p:
        raise ValueError(f"h_alpha_n needs n > p (got n={n}, p={p})")
    minimum = (n + p + 1) // 2
    return int(2 * minimum - n + 2 * (n - minimum) * alpha)


def mcd_consistency(p, alpha):
    """Consistency factor of the MCD at the normal model (Croux & Haesbroeck 1999).

    ``alpha / F_{p+2}(chi2_p^-1(alpha))``, with ``F_k`` the chi-square distribution
    function with ``k`` degrees of freedom. It makes the covariance of the
    fraction ``alpha`` of observations closest to the center consistent.

    Parameters
    ----------
    p : int
        Number of variables.
    alpha : float in (0, 1]
        Fraction of observations kept; 1 gives 1.

    Returns
    -------
    float

    Examples
    --------
    >>> from robustmodels.covariance import mcd_consistency
    >>> round(mcd_consistency(3, 0.5), 4)
    2.4574
    """
    check_count("p", p)
    check_range("alpha", alpha, 0, 1, low_open=True)
    if alpha == 1:
        return 1.0
    cutoff = chi2.ppf(alpha, p)
    return float(alpha / chi2.cdf(cutoff, p + 2))


# Pison, Van Aelst & Willems (2002): curves f(n) = 1 - exp(a) / n**b.
# (reweighted, alpha) -> [(a, b) for p = 1, (a, b) for p = 2, two (a, b) points for p > 2]
_PISON = {
    (False, 0.5): [
        (0.262024211897096, 0.604756680630497),
        (0.673292623522027, 0.691365864961895),
        (-1.42764571687802, 1.26263336932151),
        (-1.06141115981725, 1.28907991440387),
    ],
    (False, 0.875): [
        (-0.351584646688712, 1.01646567502486),
        (0.446537815635445, 1.06690782995919),
        (-0.455179464070565, 1.11192541278794),
        (-0.294241208320834, 1.09649329149811),
    ],
    (True, 0.5): [
        (1.11098143415027, 1.5182890270453),
        (3.11101712909049, 1.91401056721863),
        (-1.02842572724793, 1.67659883081926),
        (-0.26800273450853, 1.35968562893582),
    ],
    (True, 0.875): [
        (-0.66046776772861, 0.88939595831888),
        (0.79473550581058, 1.10081930350091),
        (-0.544482443573914, 1.25994483222292),
        (-0.343791072183285, 1.25159004257133),
    ],
}


def _bias_curve(p, n, alpha, reweighted):
    fit = _PISON[reweighted, alpha]
    if p <= 2:
        a, b = fit[p - 1]
    else:  # line through log(-a_j / p^b_j) at log(2 p^2) and log(3 p^2)
        (a1, b1), (a2, b2) = fit[2], fit[3]
        y1, y2 = np.log(-a1 / p**b1), np.log(-a2 / p**b2)
        b = (y1 - y2) / np.log(1.5)
        a = y1 + b * np.log(2 * p * p)
    return 1 - np.exp(a) / n**b


def mcd_small_sample(p, n, alpha, reweighted=False):
    """Small-sample factor of the MCD (Pison, Van Aelst & Willems 2002).

    The paper fits the factors at ``alpha = 0.5`` and ``0.875``; other coverages
    are interpolated linearly, reaching 1 at ``alpha = 1``.

    Parameters
    ----------
    p, n : int
        Number of variables and of observations.
    alpha : float in [0.5, 1]
        Coverage.
    reweighted : bool, default False
        Factor for the reweighted instead of the raw MCD.

    Returns
    -------
    float

    Examples
    --------
    >>> from robustmodels.covariance import mcd_small_sample
    >>> round(mcd_small_sample(3, 100, 0.5), 4)
    1.0989
    """
    check_count("p", p)
    check_count("n", n)
    check_range("alpha", alpha, 0.5, 1)
    f50 = _bias_curve(p, n, 0.5, reweighted)
    f875 = _bias_curve(p, n, 0.875, reweighted)
    if alpha <= 0.875:
        f = f50 + (f875 - f50) * (alpha - 0.5) / 0.375
    else:
        f = f875 + (1 - f875) * (alpha - 0.875) / 0.125
    return float(1 / f) if f > 0 else 1.0


class MRCD(_RobustCovariance):
    """Minimum Regularized Covariance Determinant estimator (Boudt et al. 2020).

    The data are standardized with the column medians and Qn scales. The
    ``h``-subset ``H`` then minimizes the determinant of
    ``K(H) = rho * I + (1 - rho) * c_alpha * cov(H)``, where ``c_alpha`` is the
    MCD consistency factor. The common ``rho`` is selected from the initial
    subsets by step 3.4 of Boudt et al. (2020), then held fixed during C-steps.
    The search does not certify a global minimum. In the original units
    the regularization target is the diagonal matrix of squared Qn scales.
    Works with more variables than observations; when the subset covariance is
    already well-conditioned, ``rho = 0`` and the result is the raw MCD. There is
    no reweighting step.

    With more variables than observations the starting subsets come from
    eigenvectors that are not unique, so results can differ slightly between
    platforms and linear algebra libraries.

    Parameters
    ----------
    alpha : float in [0.5, 1], default 0.5
        Coverage; the subset size is ``h = ceil(alpha * n)``.
    max_condition : float, default 50
        Condition-number bound used to calibrate the starting subsets in
        standardized coordinates. It is not a hard bound on the final subset
        or on the covariance in original units.
    rho : float in [0, 1), optional
        Fixed regularization weight, which replaces ``max_condition``. Needed when a
        starting subset has zero scatter.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Mean of the ``h``-subset.
    covariance_ : ndarray of shape (n_features, n_features)
        Regularized covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Inverse of ``covariance_``.
    support_ : ndarray of bool, shape (n_samples,)
        The ``h``-subset.
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    rho_ : float
        Regularization weight used.
    h_ : int
        Subset size.

    References
    ----------
    Boudt, K., Rousseeuw, P. J., Vanduffel, S. and Verdonck, T. (2020). The
    minimum regularized covariance determinant estimator. *Statistics and
    Computing*, 30(1), 113-128.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import MRCD
    >>> X = np.random.default_rng(0).normal(size=(20, 50))  # p > n
    >>> mrcd = MRCD().fit(X)
    >>> mrcd.covariance_.shape
    (50, 50)
    >>> 0 < mrcd.rho_ < 1
    True
    """

    def __init__(self, *, alpha=0.5, max_condition=50.0, rho=None):
        self.alpha = alpha
        self.max_condition = max_condition
        self.rho = rho

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        check_range("alpha", self.alpha, 0.5, 1)
        check_range("max_condition", self.max_condition, 1, np.inf, low_open=True)
        if self.rho is not None:
            check_range("rho", self.rho, 0, 1, high_open=True)
        n, p = X.shape
        h = math.ceil(self.alpha * n)
        if h < 2:
            raise ValueError("MRCD needs at least two observations in its subset")
        center, scale = _center_scale(X)
        standardized = (X - center) / scale
        correction = mcd_consistency(p, h / n)
        starts, rho = self._starting_subsets(standardized, h, correction)
        try:
            location, covariance, subset = _best_regularized_subset(
                standardized, starts, h, rho, correction
            )
        except np.linalg.LinAlgError:
            raise ValueError("MRCD covariance is singular; pass a larger rho") from None
        self.h_, self.rho_ = h, float(rho)
        self.support_ = np.isin(np.arange(n), subset)
        location = center + scale * location
        covariance = np.outer(scale, scale) * covariance
        self._set_estimates(location, covariance, X)
        return self

    def _starting_subsets(self, X, h, correction):
        if h == len(X):
            starts = [np.arange(h)]
        else:
            starts = _deterministic_starts(X, h)
        if self.rho is not None:
            return starts, self.rho
        weights = []
        for subset in starts:
            covariance = correction * _mean_cov(X[subset])[1]
            weights.append(_min_rho(covariance, self.max_condition))
        weights = np.array(weights)
        # Boudt et al. (2020), algorithm step 3.4. The median need not
        # coincide with one of the six weights: do not round it down.
        rho = weights.max() if weights.max() <= 0.1 else max(0.1, np.median(weights))
        selected = [subset for subset, weight in zip(starts, weights, strict=True) if weight <= rho]
        return selected, rho


def _min_rho(covariance, max_condition):
    """Smallest rho for cond(rho I + (1 - rho) covariance) <= max_condition."""
    eigenvalues = np.linalg.eigvalsh(covariance)
    smallest = max(eigenvalues[0], 0.0)
    largest = eigenvalues[-1]
    if largest <= 0:
        raise ValueError("MRCD subset has zero scatter; pass a positive rho")
    if largest <= max_condition * smallest:
        return 0.0
    numerator = largest - max_condition * smallest
    denominator = max_condition * (1 - smallest) - (1 - largest)
    rho = numerator / denominator
    if rho >= 1:
        raise ValueError("MRCD regularization is too close to 1; rescale the data")
    return rho


def _regularized_estimates(X, subset, rho, correction):
    location, covariance = _mean_cov(X[subset])
    target = np.eye(X.shape[1])
    covariance = rho * target + (1 - rho) * correction * covariance
    return location, covariance


def _best_regularized_subset(X, starts, h, rho, correction):
    candidates = [_mrcd_steps(X, subset, h, rho, correction) for subset in starts]
    _, subset = min(candidates, key=lambda candidate: candidate[0])
    location, covariance = _regularized_estimates(X, subset, rho, correction)
    return location, covariance, subset


def _mrcd_steps(X, subset, h, rho, correction, max_steps=200):
    """Concentration steps with the same regularization weight at every step."""
    for _ in range(max_steps):  # pragma: no branch (C-steps stop long before the cap)
        location, covariance = _regularized_estimates(X, subset, rho, correction)
        distances = _mahalanobis_sq(X, location, covariance)
        updated = np.sort(np.argpartition(distances, h - 1)[:h])
        if np.array_equal(updated, subset):
            break
        subset = updated
    _, covariance = _regularized_estimates(X, subset, rho, correction)
    return np.linalg.slogdet(covariance)[1], subset


class MVE(_RobustCovariance):
    """Minimum Volume Ellipsoid estimator (Rousseeuw 1985).

    Approximation to the center and scaled shape of the smallest ellipsoid
    that covers ``h`` observations. The search draws random ``(p + 1)``-subsets,
    then refits the 10 best ellipsoids to the ``h`` points they cover while the
    volume decreases. This heuristic does not certify a minimum-volume
    ellipsoid, even on each selected subset. The shape is scaled for consistency
    at the normal model and reweighted as in :class:`MCD`. Breakdown point about 50% for
    ``alpha = 0.5``. Needs ``n > p + 1``.

    Parameters
    ----------
    alpha : float in [0.5, 1), default 0.5
        Coverage, as for :class:`MCD`. Full coverage has no finite scale correction.
    n_trials : int, default 500
        Random starting subsets.
    reweighting : bool, default True
        Recompute mean and covariance from the observations whose squared robust
        distance is below the 0.975 chi-square quantile.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the random subsets.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Final location.
    covariance_ : ndarray of shape (n_features, n_features)
        Final covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Pseudo-inverse of ``covariance_``.
    support_ : ndarray of bool, shape (n_samples,)
        Observations used for the final estimates.
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    raw_location_ : ndarray of shape (n_features,)
        Center of the best ellipsoid.
    raw_covariance_ : ndarray of shape (n_features, n_features)
        Scaled shape of the best ellipsoid.
    raw_support_ : ndarray of bool, shape (n_samples,)
        Observations inside the best ellipsoid.
    h_ : int
        Number of observations covered.

    References
    ----------
    Rousseeuw, P. J. (1985). Multivariate estimation with high breakdown point.
    In W. Grossmann et al. (eds.), *Mathematical Statistics and Applications*,
    Vol. B, 283-297. Reidel.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import MVE
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:10] += 10
    >>> mve = MVE(random_state=0).fit(X)
    >>> mve.location_.round(2).tolist()
    [0.01, 0.1]
    """

    def __init__(self, *, alpha=0.5, n_trials=500, reweighting=True, random_state=None):
        self.alpha = alpha
        self.n_trials = n_trials
        self.reweighting = reweighting
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        check_range("alpha", self.alpha, 0.5, 1, high_open=True)
        n, p = X.shape
        if n <= p + 1:
            raise ValueError(f"MVE needs n > p + 1 (got n={n}, p={p})")
        h = h_alpha_n(self.alpha, n, p)
        check_count("n_trials", self.n_trials)
        _, location, shape, distances = self._best_ellipsoid(X, h)
        radius_squared = np.partition(distances, h - 1)[h - 1]
        self.h_ = h
        self.raw_location_ = location
        self.raw_covariance_ = shape * radius_squared / chi2.ppf(h / n, p)
        self.raw_support_ = distances <= radius_squared
        covariance = self.raw_covariance_
        self.support_ = self.raw_support_
        if self.reweighting:
            location, covariance, self.support_ = _reweight(X, location, covariance)
        self._set_estimates(location, covariance, X)
        return self

    def _best_ellipsoid(self, X, h):
        rng = as_generator(self.random_state)
        candidates = []
        for _ in range(self.n_trials):
            subset = _random_subset(X, rng)
            if subset is None:
                raise ValueError("the data lie on a hyperplane")
            location, covariance = _mean_cov(X[subset])
            candidates.append(_ellipsoid(X, location, covariance, h))
        candidates.sort(key=lambda candidate: candidate[0])
        refined = [_refine_ellipsoid(X, candidate, h) for candidate in candidates[:10]]
        return min(refined, key=lambda candidate: candidate[0])


def _ellipsoid(X, location, covariance, h):
    """Return log-volume, location, shape and distances for an h-point ellipsoid."""
    distances = _mahalanobis_sq(X, location, covariance)
    radius_squared = np.partition(distances, h - 1)[h - 1]
    volume = _logdet(covariance) + len(covariance) * np.log(radius_squared)
    return volume, location, covariance, distances


def _refine_ellipsoid(X, candidate, h):
    """Refit the covered h points while the ellipsoid volume decreases."""
    for _ in range(100):  # pragma: no branch (volumes stop decreasing long before the cap)
        volume, _, _, distances = candidate
        subset = np.argpartition(distances, h - 1)[:h]
        location, covariance = _mean_cov(X[subset])
        if _logdet(covariance) == -np.inf:
            break
        updated = _ellipsoid(X, location, covariance, h)
        if updated[0] >= volume:
            break
        candidate = updated
    return candidate


_SCALES = {"tau": tau_scale, "qn": qn, "mad": mad, "pn": pn}


class OGK(_RobustCovariance):
    """Orthogonalized Gnanadesikan-Kettenring estimator (Maronna & Zamar 2002).

    Robust pairwise covariances ``(s(X + Y)^2 - s(X - Y)^2) / 4``, with ``s`` a
    robust univariate scale, made positive definite by orthogonalization.
    Deterministic and fast for many variables. It is equivariant under shifts
    and rescaling of the variables, but not under rotations.

    Parameters
    ----------
    scale : {"tau", "qn", "mad", "pn"}, default "tau"
        Univariate scale. "tau" is centered at the tau location, the others at
        the median.
    n_iter : int, default 2
        Orthogonalization iterations.
    reweighting : bool, default True
        Recompute mean and covariance from the observations with
        ``d^2 <= chi2_p^-1(beta) * median(d^2) / chi2_p^-1(0.5)``. The reweighted
        covariance is corrected for consistency at the normal model, which R's
        ``covOGK`` omits.
    beta : float in (0, 1), default 0.9
        Quantile of the reweighting cutoff.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Final location.
    covariance_ : ndarray of shape (n_features, n_features)
        Final covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Pseudo-inverse of ``covariance_``.
    support_ : ndarray of bool, shape (n_samples,)
        Observations used for the final estimates (all if ``reweighting=False``).
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    raw_location_ : ndarray of shape (n_features,)
        Location before reweighting.
    raw_covariance_ : ndarray of shape (n_features, n_features)
        Covariance before reweighting.

    References
    ----------
    Maronna, R. A. and Zamar, R. H. (2002). Robust estimates of location and
    dispersion for high-dimensional datasets. *Technometrics*, 44(4), 307-317.

    Gnanadesikan, R. and Kettenring, J. R. (1972). Robust estimates, residuals,
    and outlier detection with multiresponse data. *Biometrics*, 28(1), 81-124.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import OGK
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:10] += 10
    >>> OGK().fit(X).covariance_.round(1).tolist()
    [[1.1, 0.2], [0.2, 1.1]]
    """

    def __init__(self, *, scale="tau", n_iter=2, reweighting=True, beta=0.9):
        self.scale = scale
        self.n_iter = n_iter
        self.reweighting = reweighting
        self.beta = beta

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        check_option("scale", self.scale, _SCALES)
        check_count("n_iter", self.n_iter)
        check_range("beta", self.beta, 0, 1, low_open=True, high_open=True)
        scale_fn = _SCALES[self.scale]
        transformed, basis = _orthogonalize(X, scale_fn, self.n_iter)
        if self.scale == "tau":
            center = np.array([_tau_estimates(column, 4.5)[0] for column in transformed.T])
        else:
            center = np.median(transformed, axis=0)
        scales = _scales(transformed, scale_fn)
        distances = np.sum(((transformed - center) / scales) ** 2, axis=1)
        self.raw_location_ = basis @ center
        self.raw_covariance_ = (basis * scales**2) @ basis.T
        location, covariance = self._reweighted_estimates(X, distances)
        self._set_estimates(location, covariance, X)
        return self

    def _reweighted_estimates(self, X, distances):
        self.support_ = np.ones(len(X), dtype=bool)
        if not self.reweighting:
            return self.raw_location_, self.raw_covariance_
        p = X.shape[1]
        cutoff = np.median(distances) * chi2.ppf(self.beta, p) / chi2.ppf(0.5, p)
        self.support_ = distances <= cutoff
        if not self.support_.any():
            raise ValueError("beta is too small: reweighting selects no observations")
        selected = X[self.support_]
        location = selected.mean(axis=0)
        centered = selected - location
        covariance = centered.T @ centered / len(selected)
        covariance *= mcd_consistency(p, self.beta)
        return location, covariance


def _orthogonalize(X, scale_fn, n_iter):
    """Rotate standardized columns and record how to return to the original basis."""
    transformed = X
    basis = np.eye(X.shape[1])
    for _ in range(n_iter):
        scales = _scales(transformed, scale_fn)
        standardized = transformed / scales
        eigenvectors = np.linalg.eigh(_gk_covariance(standardized, scale_fn))[1]
        transformed = standardized @ eigenvectors
        basis = basis @ (scales[:, None] * eigenvectors)
    return transformed, basis


def _scales(X, scale_fn):
    scales = np.array([scale_fn(column) for column in X.T])
    if np.any(scales <= 0):
        raise ValueError("a variable has zero robust scale (more than half of its values tied)")
    return scales


class StahelDonoho(_RobustCovariance):
    """Stahel-Donoho estimator of location and scatter (Stahel 1981; Donoho 1982).

    Weighted mean and covariance with weights ``w(r_i)`` that decrease with the
    Stahel-Donoho outlyingness ``r_i`` of each observation, computed by
    :class:`~robustmodels.outliers.StahelDonohoOutlyingness`. The weights are
    ``w(r) = min(1, (c / r)^2)`` with ``c`` the square root of a chi-square
    quantile, as recommended by Maronna & Yohai (1995). The covariance is then
    scaled so that the median squared distance equals the chi-square median,
    which makes it consistent at the normal model. With all directions the
    estimator is affine equivariant and its breakdown point approaches 50%; a
    finite set of directions approximates it. An outlier keeps a small bounded
    influence, since ``w(r) r^2`` equals ``c^2`` beyond ``c``. Needs ``n > p``.

    Parameters
    ----------
    quantile : float in (0, 1), default 0.95
        Chi-square quantile for ``c``: observations with outlyingness up to
        ``c`` keep weight 1.
    n_directions : int, default None
        Number of directions; ``250 * n_features`` by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Weighted mean.
    covariance_ : ndarray of shape (n_features, n_features)
        Scaled weighted covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Pseudo-inverse of ``covariance_``.
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    outlyingness_ : ndarray of shape (n_samples,)
        Stahel-Donoho outlyingness of the training data.
    weights_ : ndarray of shape (n_samples,)
        Weights in [0, 1].

    Notes
    -----
    R's ``rrcov::CovSde`` uses a different, redescending weight function, so
    its estimates differ on contaminated data.

    References
    ----------
    Stahel, W. A. (1981). *Robuste Schätzungen: infinitesimale Optimalität und
    Schätzungen von Kovarianzmatrizen*. PhD thesis, ETH Zürich.

    Donoho, D. L. (1982). *Breakdown properties of multivariate location
    estimators*. Qualifying paper, Harvard University.

    Maronna, R. A. and Yohai, V. J. (1995). The behavior of the Stahel-Donoho
    robust multivariate estimator. *Journal of the American Statistical
    Association*, 90(429), 330-341.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import StahelDonoho
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:10] += 10
    >>> sd = StahelDonoho(random_state=0).fit(X)
    >>> sd.location_.round(1).tolist()
    [0.0, 0.2]
    >>> bool(sd.weights_[:10].max() < 0.1)  # the outliers hardly count
    True
    """

    def __init__(self, *, quantile=0.95, n_directions=None, random_state=None):
        self.quantile = quantile
        self.n_directions = n_directions
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        check_range("quantile", self.quantile, 0, 1, low_open=True, high_open=True)
        detector = StahelDonohoOutlyingness(
            n_directions=self.n_directions, random_state=self.random_state
        ).fit(X)
        p = X.shape[1]
        self.outlyingness_ = detector.outlyingness_
        with np.errstate(divide="ignore"):
            ratio = np.sqrt(chi2.ppf(self.quantile, p)) / self.outlyingness_
        self.weights_ = np.minimum(1, ratio**2)
        location = self.weights_ @ X / self.weights_.sum()
        centered = X - location
        covariance = (self.weights_[:, None] * centered).T @ centered / self.weights_.sum()
        if _logdet(covariance) == -np.inf:
            raise ValueError("the weighted observations lie on a hyperplane")
        covariance *= np.median(_mahalanobis_sq(X, location, covariance)) / chi2.ppf(0.5, p)
        self._set_estimates(location, covariance, X)
        return self


class SEstimator(_RobustCovariance):
    """S-estimator of location and scatter (Davies 1987), with Tukey's bisquare.

    Finds the location ``mu`` and the shape ``Gamma`` (determinant 1) that
    minimize the M-scale ``s`` of the distances
    ``d_i = sqrt((x_i - mu)' Gamma^-1 (x_i - mu))``, where ``s`` solves
    ``mean(rho(d_i / s)) = breakdown``. The covariance is ``s^2 * Gamma``,
    consistent at the normal model. Computed with the FAST-S algorithm
    (Salibián-Barrera, Van Aelst & Willems 2006), without a global-minimum
    certificate. The theoretical estimator is affine equivariant; needs
    ``n > p``.

    Parameters
    ----------
    breakdown : float in (0, 0.5], default 0.5
        Breakdown point.
    n_trials : int, default 500
        Random ``(p + 1)``-subsets, each refined with ``k_steps`` I-steps.
    n_best : int, default 5
        Best candidates iterated to convergence.
    k_steps : int, default 2
        I-steps applied to each random subset.
    max_iter : int, default 200
        Maximum I-steps for each of the best candidates.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the random subsets.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Location ``mu``.
    covariance_ : ndarray of shape (n_features, n_features)
        Covariance matrix ``s^2 * Gamma``.
    precision_ : ndarray of shape (n_features, n_features)
        Pseudo-inverse of ``covariance_``.
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    scale_ : float
        The minimized M-scale ``s``.

    References
    ----------
    Davies, P. L. (1987). Asymptotic behaviour of S-estimates of multivariate
    location parameters and dispersion matrices. *Annals of Statistics*, 15(3),
    1269-1292.

    Salibián-Barrera, M., Van Aelst, S. and Willems, G. (2006). Principal
    components analysis based on multivariate MM estimators with fast and robust
    bootstrap. *Journal of the American Statistical Association*, 101(475),
    1198-1211.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import SEstimator
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:10] += 10
    >>> s = SEstimator(random_state=0).fit(X)
    >>> s.covariance_.round(1).tolist()
    [[1.1, 0.1], [0.1, 1.3]]
    """

    def __init__(
        self, *, breakdown=0.5, n_trials=500, n_best=5, k_steps=2, max_iter=200, random_state=None
    ):
        self.breakdown = breakdown
        self.n_trials = n_trials
        self.n_best = n_best
        self.k_steps = k_steps
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        location, shape, self.scale_ = self._fit_s(X)
        self._set_estimates(location, self.scale_**2 * shape, X)
        return self

    def _fit_s(self, X):
        """Returns (location, shape with det 1, scale)."""
        n, p = X.shape
        for name in ("n_trials", "n_best", "k_steps", "max_iter"):
            check_count(name, getattr(self, name))
        if n <= p:
            raise ValueError(f"S-estimation needs n > p (got n={n}, p={p})")
        c = _tuning_for_breakdown(self.breakdown, p)
        rng = as_generator(self.random_state)
        candidates = self._initial_candidates(X, c, rng)
        candidates.sort(key=lambda candidate: candidate[2])
        refined = []
        for location, shape, _ in candidates[: self.n_best]:
            refined.append(_s_steps(X, location, shape, c, self.breakdown, self.max_iter))
        return min(refined, key=lambda candidate: candidate[2])

    def _initial_candidates(self, X, c, rng):
        candidates = []
        for _ in range(self.n_trials):
            subset = _random_subset(X, rng)
            if subset is None:
                raise ValueError("the data lie on a hyperplane")
            location, covariance = _mean_cov(X[subset])
            shape = _unit_det(covariance)
            candidates.append(_s_steps(X, location, shape, c, self.breakdown, self.k_steps))
        return candidates


def _unit_det(covariance):
    """Rescale a positive definite covariance matrix to determinant 1."""
    geometric_mean = np.exp(_logdet(covariance) / len(covariance))
    return covariance / geometric_mean


def _s_steps(X, location, shape, c, breakdown, n_steps, tol=1e-10):
    """Reweight location and shape while the M-scale decreases."""
    distances = np.sqrt(_mahalanobis_sq(X, location, shape))
    scale = _mscale_bisquare(distances, c, breakdown)
    for _ in range(n_steps):
        if scale == 0:
            break
        weights = _bisquare_weights(distances / scale, c)
        new_location, new_shape = _weighted_shape(X, weights)
        new_distances = np.sqrt(_mahalanobis_sq(X, new_location, new_shape))
        new_scale = _mscale_bisquare(new_distances, c, breakdown, s0=scale)
        if new_scale > scale:  # pragma: no cover (rounding only: S-steps decrease the scale)
            break
        converged = scale - new_scale <= tol * scale
        location, shape = new_location, new_shape
        distances, scale = new_distances, new_scale
        if converged:
            break
    return location, shape, scale


def _weighted_shape(X, weights):
    """Weighted mean and weighted scatter rescaled to determinant 1."""
    total = weights.sum()
    if total == 0:
        raise ValueError("all observations have zero bisquare weight")
    location = weights @ X / total
    centered = X - location
    scatter = (weights[:, None] * centered).T @ centered
    if _logdet(scatter) == -np.inf:
        raise ValueError("bisquare weights produce a singular scatter matrix")
    return location, _unit_det(scatter)


class MMEstimator(_RobustCovariance):
    """MM-estimator of location and scatter (Tatsuoka & Tyler 2000).

    Starts from a bisquare :class:`SEstimator` and keeps its scale ``s``. A
    bisquare M-estimator of location and shape, tuned for the requested Gaussian
    efficiency of the shape, then refines the estimates. It keeps the breakdown
    point of the S-estimator and gains efficiency. Affine equivariant; needs
    ``n > p``.

    Parameters
    ----------
    breakdown : float in (0, 0.5], default 0.5
        Breakdown point of the initial S-estimator.
    efficiency : float in (0, 1), default 0.95
        Asymptotic Gaussian efficiency of the shape, not of the full covariance
        or location. Must imply a final bisquare tuning constant at least as
        large as the initial S tuning, to preserve the breakdown guarantee;
        incompatible combinations of efficiency, breakdown and dimension raise
        ``ValueError``.
    n_trials : int, default 500
        Random subsets of the initial S-estimator.
    max_iter : int, default 200
        Maximum iterations of the M-step.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the initial S-estimator.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        MM location.
    covariance_ : ndarray of shape (n_features, n_features)
        MM covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Pseudo-inverse of ``covariance_``.
    dist_ : ndarray of shape (n_samples,)
        Squared robust distances of the training data.
    scale_ : float
        Scale of the S-estimator.
    s_location_ : ndarray of shape (n_features,)
        Location of the S-estimator.
    s_covariance_ : ndarray of shape (n_features, n_features)
        Covariance of the S-estimator.

    References
    ----------
    Tatsuoka, K. S. and Tyler, D. E. (2000). On the uniqueness of S-functionals
    and M-functionals under nonelliptical distributions. *Annals of Statistics*,
    28(4), 1219-1243.

    Salibián-Barrera, M., Van Aelst, S. and Willems, G. (2006). Principal
    components analysis based on multivariate MM estimators with fast and robust
    bootstrap. *Journal of the American Statistical Association*, 101(475),
    1198-1211.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.covariance import MMEstimator
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:10] += 10
    >>> mm = MMEstimator(random_state=0).fit(X)
    >>> mm.covariance_.round(1).tolist()
    [[1.2, 0.1], [0.1, 1.2]]
    """

    def __init__(
        self, *, breakdown=0.5, efficiency=0.95, n_trials=500, max_iter=200, random_state=None
    ):
        self.breakdown = breakdown
        self.efficiency = efficiency
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = self._validate(X)
        check_count("max_iter", self.max_iter)
        c = _tuning_for_shape_efficiency(self.efficiency, X.shape[1])
        c0 = _tuning_for_breakdown(self.breakdown, X.shape[1])
        if c < c0:
            raise ValueError(
                "efficiency is too low for this breakdown and dimension: "
                "MM requires the final bisquare tuning constant >= the initial one"
            )
        initial = SEstimator(
            breakdown=self.breakdown, n_trials=self.n_trials, random_state=self.random_state
        )
        location, shape, scale = initial._fit_s(X)
        self.scale_ = scale
        self.s_location_ = location
        self.s_covariance_ = scale**2 * shape
        location, shape = _refine_shape(X, location, shape, scale, c, self.max_iter)
        self._set_estimates(location, scale**2 * shape, X)
        return self


def _refine_shape(X, location, shape, scale, c, max_iter):
    """Decrease bisquare loss while keeping the initial S-scale fixed."""
    if scale == 0:
        return location, shape
    distances = np.sqrt(_mahalanobis_sq(X, location, shape))
    loss = np.mean(_rho(distances / scale, c))
    for _ in range(max_iter):
        weights = _bisquare_weights(distances / scale, c)
        new_location, new_shape = _weighted_shape(X, weights)
        new_distances = np.sqrt(_mahalanobis_sq(X, new_location, new_shape))
        new_loss = np.mean(_rho(new_distances / scale, c))
        if new_loss > loss:  # pragma: no cover (rounding only: IRLS decreases the loss)
            break
        converged = loss - new_loss <= 1e-12
        location, shape, distances, loss = new_location, new_shape, new_distances, new_loss
        if converged:
            break
    return location, shape


def _mahalanobis_sq(X, location, cov):
    """Squared Mahalanobis distances (``cov`` must be positive definite)."""
    centered = X - location
    cholesky = np.linalg.cholesky(cov)
    standardized = np.linalg.solve(cholesky, centered.T)
    return np.sum(standardized**2, axis=0)


def _logdet(cov):
    """``log det(cov)``, or ``-inf`` if ``cov`` is numerically singular."""
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        return -np.inf
    return float(np.log(eigenvalues).sum())


def _mean_cov(X):
    return X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))


def _center_scale(X):
    """Column medians and Qn scales (MAD, then 1, when a scale is zero)."""
    scale = np.array([_column_scale(column) for column in X.T])
    return np.median(X, axis=0), scale


def _column_scale(column):
    scale = qn(column)
    if scale == 0:
        scale = mad(column)
    if scale == 0:
        scale = 1.0
    return scale


def _random_subset(X, rng):
    """Random (p+1)-subset, enlarged until its covariance is non-singular (None if impossible)."""
    n, p = X.shape
    order = rng.permutation(n)
    for k in range(p + 1, n + 1):
        subset = order[:k]
        _, covariance = _mean_cov(X[subset])
        if _logdet(covariance) > -np.inf:
            return subset
    return None


def _reweight(X, location, cov, quantile=0.975):
    """Mean and covariance of the points with ``d^2 < chi2_p(quantile)``.

    The covariance is made consistent for that cutoff (not for the fraction
    actually kept, which outliers would distort). If every point is kept, the
    sample covariance is returned unchanged, as in robustbase.
    """
    p = X.shape[1]
    keep = _mahalanobis_sq(X, location, cov) < chi2.ppf(quantile, p)
    if keep.sum() <= p:
        return location, cov, keep
    loc, cov = _mean_cov(X[keep])
    if not keep.all():
        cov = cov * mcd_consistency(p, quantile)
    return loc, cov, keep


def _gk_covariance(X, scale_fn):
    """Pairwise Gnanadesikan-Kettenring scatter with a unit diagonal."""
    matrix = np.eye(X.shape[1])
    for i in range(X.shape[1]):
        for j in range(i):
            a, b = X[:, i], X[:, j]
            sum_variance = scale_fn(a + b) ** 2
            difference_variance = scale_fn(a - b) ** 2
            covariance = (sum_variance - difference_variance) / 4
            matrix[i, j] = matrix[j, i] = covariance
    return matrix
