"""Rank-based regression and exact rank tests that condition on ties.

:class:`RankRegressor` is a scikit-learn regressor; ``fit(X, y)`` sets
``coef_``, ``intercept_``, the scale parameters of its inference and
coefficient tables, and ``predict(X)`` returns ``X @ coef_ + intercept_``.
:func:`rank_sum_test` and :func:`signed_rank_test` compute exact p-values from
the permutation distribution of midrank scores given the observed ties, which
remains valid with ties and zeros.
"""

import math
from fractions import Fraction
from numbers import Real
from typing import NamedTuple

import numpy as np
from scipy import linalg, optimize, stats
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_1d, check_option, is_exact_fit
from .regression import _coefficient_table
from .scale import _count, _kth_smallest, _pairwise_rows, mad

__all__ = [
    "RankRegressor",
    "RankTestResult",
    "drop_test",
    "rank_anova",
    "rank_sum_test",
    "signed_rank_test",
]

_ALTERNATIVES = ("two-sided", "less", "greater")
_TAU_IMPLICIT_MIN_SIZE = 2048  # condensed distances are faster for small samples


class RankRegressor(RegressorMixin, BaseEstimator):
    """Rank-based (R) estimator of linear regression with Wilcoxon scores (Jaeckel 1972).

    The slopes minimize Jaeckel's dispersion ``D(b) = sum(a(R_i) e_i)`` of the
    residuals ``e = y - X b``, with ``R_i`` the rank of ``e_i`` and Wilcoxon
    scores ``a(i) = sqrt(12) (i - (n + 1) / 2) / sqrt(n (n - 1))``, which Rfit
    standardizes so that ``sum(a(i)^2) = n + 1``. As ``D`` equals
    ``sqrt(3 / (n (n - 1)))`` times the sum of ``|e_i - e_j|`` over pairs, the
    fit is an L1 regression on the pairwise differences, solved exactly by
    linear programming (Rfit approximates it by quasi-Newton steps). ``D``
    ignores the intercept, which is the median of the residuals. The Gaussian
    efficiency of the slopes is ``3 / pi = 95.5%``, and they are more efficient
    than least squares for heavy tails; they resist outliers in ``y`` but not
    leverage points in ``X``. The covariance of the coefficients
    is ``tau^2 (X'X)^-1`` for the centred slopes and adds ``tau_s^2 / n`` to
    the variance of the intercept, with ``tau`` estimated as Koul, Sievers &
    McKean (1987) and ``tau_s`` from the confidence interval of the median.
    Score standardization and small-sample corrections are Rfit's: in
    simulations they gave 95% intervals and 5% tests closer to their nominal
    levels than the unstandardized scores. Uses the
    ``n (n - 1) / 2`` pairs of observations, so memory grows as ``n^2``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slopes.
    intercept_ : float
        Median of ``y - X @ coef_``.
    scale_ : float
        MAD of the residuals, which standardizes them in the diagnostic plots.
    dispersion_ : float
        Jaeckel's dispersion of the residuals at the fit.
    tau_ : float
        Scale parameter of the slopes; ``1 / (sqrt(12) integral f^2)`` for
        error density ``f``.
    tau_s_ : float
        Scale parameter of the intercept, ``1 / (2 f(median))``.
    cov_params_ : ndarray of shape (n_features + 1, n_features + 1)
        Asymptotic covariance of ``[intercept_, *coef_]``.
    std_errors_, t_values_, p_values_ : ndarray of shape (n_features + 1,)
        Standard errors, t statistics and two-sided p-values from a t
        distribution with ``n - p - 1`` degrees of freedom, as Rfit.

    References
    ----------
    Jaeckel, L. A. (1972). Estimating regression coefficients by minimizing
    the dispersion of the residuals. *The Annals of Mathematical Statistics*,
    43(5), 1449-1458.

    Koul, H. L., Sievers, G. L. and McKean, J. W. (1987). An estimator of the
    scale parameter for the rank analysis of linear models under general score
    functions. *Scandinavian Journal of Statistics*, 14(2), 131-141.

    Hettmansperger, T. P. and McKean, J. W. (2011). *Robust Nonparametric
    Statistical Methods*, 2nd edition. CRC Press.

    Kloke, J. D. and McKean, J. W. (2012). Rfit: Rank-based estimation for
    linear models. *The R Journal*, 4(2), 57-64.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.rank import RankRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> y = 1 + X @ [2.0, -1.0] + rng.standard_t(3, size=100)
    >>> y[:5] += 20  # five outliers move least squares to (1.95, 2.3, -1.34)
    >>> fit = RankRegressor().fit(X, y)
    >>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
    (0.85, [2.14, -1.17])
    """

    fit_intercept = True  # read by the coefficient table

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True, ensure_min_samples=2)
        n, p = X.shape
        if n <= p + 1:
            raise ValueError(f"needs more observations than coefficients (got n={n}, p={p + 1})")
        centred = X - X.mean(axis=0)
        if np.linalg.matrix_rank(centred) < p:
            raise ValueError("X is singular: its columns (and the intercept) are collinear")
        self._design_basis = np.linalg.qr(centred, mode="reduced")[0]
        self._response = y.copy()
        self.coef_ = _wilcoxon_slopes(centred, y)
        residuals = y - X @ self.coef_
        self.intercept_ = float(np.median(residuals))
        residuals = residuals - self.intercept_
        self.scale_ = mad(residuals)
        self.dispersion_ = _dispersion(residuals)
        self._null_dispersion, self._n_samples = _dispersion(y), n
        exact = is_exact_fit(y, residuals)
        self.tau_ = np.nan if exact else _tau(residuals, p)
        self.tau_s_ = np.nan if exact else _tau_s(residuals, p + 1)
        if np.isfinite(self.tau_) and self.tau_s_ > 0:
            design = np.column_stack([np.ones(n), X])
            cov = self.tau_**2 * linalg.pinvh(design.T @ design)
            cov[0, 0] += (self.tau_s_**2 - self.tau_**2) / n
        else:
            cov = np.full((p + 1, p + 1), np.nan)
        self.cov_params_ = cov
        self.std_errors_ = np.sqrt(np.diag(cov))
        self.t_values_ = np.r_[self.intercept_, self.coef_] / self.std_errors_
        self.p_values_ = 2 * stats.t.sf(np.abs(self.t_values_), n - p - 1)
        return self

    def predict(self, X):
        """Predicted values ``X @ coef_ + intercept_``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_

    def summary(self, feature_names=None):
        """Coefficient table and test of all slopes, as R's ``summary(rfit)``.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names of the columns of ``X``; ``x0, x1, ...`` by default.

        Returns
        -------
        str
            Estimates, standard errors, t values and two-sided p-values, then
            the drop in dispersion test that all slopes are zero (see
            :func:`drop_test`) and its robust R².
        """
        check_is_fitted(self)
        table = _coefficient_table(self, feature_names, "t", self.t_values_)
        test = drop_test(self)
        q, df = test.df
        r2 = q * test.statistic / (df + q * test.statistic)
        return (
            f"{table}\nDrop in dispersion test: F = {test.statistic:.4g} on {q} and {df} "
            f"degrees of freedom, p-value {test.pvalue:.3g}\nRobust R-squared: {r2:.4g}"
        )


class DropTestResult(NamedTuple):
    """Result of :func:`drop_test`."""

    statistic: float
    df: tuple
    pvalue: float


def drop_test(full, reduced=None):
    """Drop in dispersion test of nested rank regressions (McKean & Hettmansperger 1976).

    ``F = ((D_reduced - D_full) / q) / (tau / 2)``, with ``D`` the Jaeckel
    dispersions of both fits, ``q`` the number of slopes dropped and ``tau``
    from the full model, compared with an F distribution on ``q`` and
    ``n - p - 1`` degrees of freedom, as Rfit's ``drop.test``. It is the rank
    analogue of the F test of nested linear models, with approximate
    calibration for independent, identically distributed continuous errors.
    With an unidentified residual scale, the statistic and p-value are NaN.

    Parameters
    ----------
    full : fitted RankRegressor
    reduced : fitted RankRegressor, optional
        Fitted to the same ``y`` with a subset of the columns of the full
        model (or of their linear combinations). By default, the model with
        no slopes: the test that all slopes are zero.

    Returns
    -------
    DropTestResult
        ``statistic``, ``df`` (``(q, n - p - 1)``) and ``pvalue``.

    References
    ----------
    McKean, J. W. and Hettmansperger, T. P. (1976). Tests of hypotheses based
    on ranks in the general linear model. *Communications in Statistics -
    Theory and Methods*, 5(8), 693-709.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.rank import RankRegressor, drop_test
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2 * X[:, 0] + rng.standard_t(3, size=60)  # the last two columns do not matter
    >>> full, reduced = RankRegressor().fit(X, y), RankRegressor().fit(X[:, :1], y)
    >>> round(drop_test(full, reduced).pvalue, 2), bool(drop_test(full).pvalue < 1e-6)
    (0.7, True)
    """
    check_is_fitted(full)
    n, p = full._n_samples, full.n_features_in_
    if reduced is None:
        dropped, q = full._null_dispersion, p
    else:
        check_is_fitted(reduced)
        if not np.array_equal(reduced._response, full._response):
            raise ValueError("full and reduced must be fitted to the same response")
        dropped, q = reduced.dispersion_, p - reduced.n_features_in_
        if q < 1:
            raise ValueError("the reduced model must have fewer columns than the full model")
        Q, R = full._design_basis, reduced._design_basis
        if not np.allclose(R, Q @ (Q.T @ R), atol=1e-8, rtol=0):
            raise ValueError("the reduced model is not nested in the full model")
    statistic = max(dropped - full.dispersion_, 0.0) / q / (full.tau_ / 2)
    df = n - p - 1
    return DropTestResult(float(statistic), (q, df), float(stats.f.sf(statistic, q, df)))


def rank_anova(y, groups):
    """Rank-based one-way analysis of variance, as Rfit's ``oneway.rfit``.

    Fits the one-way model ``y = mu_g + e`` with :class:`RankRegressor` on
    indicator columns of the groups and tests that all group centers are equal
    with :func:`drop_test`. Unlike the Kruskal-Wallis test it estimates the
    group effects and their standard errors, and it extends to other designs
    through :class:`RankRegressor` and :func:`drop_test`.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Response.
    groups : array-like of shape (n_samples,)
        Group of each observation; at least two groups.

    Returns
    -------
    DropTestResult
        ``statistic``, ``df`` (``(k - 1, n - k)``) and ``pvalue``.

    References
    ----------
    Hettmansperger, T. P. and McKean, J. W. (2011). *Robust Nonparametric
    Statistical Methods*, 2nd edition, Section 4.2. CRC Press.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.rank import rank_anova
    >>> rng = np.random.default_rng(0)
    >>> groups = np.repeat(["a", "b", "c"], 15)
    >>> y = rng.standard_t(3, size=45) + np.repeat([0.0, 0.0, 1.5], 15)
    >>> result = rank_anova(y, groups)
    >>> result.df, round(result.pvalue, 4)
    ((2, 42), 0.0081)
    """
    y = as_1d(y)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError(f"groups has shape {groups.shape}, y has {y.shape}")
    levels, codes = np.unique(groups, return_inverse=True)
    if len(levels) < 2:
        raise ValueError("rank_anova needs at least two groups")
    indicators = (codes[:, None] == np.arange(1, len(levels))).astype(float)
    return drop_test(RankRegressor().fit(indicators, y))


def _wilcoxon_slopes(X, y, near_pairs=2):
    """Slopes minimizing the sum of ``|(y_i - y_j) - (x_i - x_j)' b|`` over pairs ``i < j``.

    At a pilot estimate, pairs whose residuals are far apart keep their signs
    near the minimum, so their terms add up to a linear function computed from
    ranks. The linear program needs only the ``near_pairs * n`` closest pairs
    (the preprocessing of Portnoy & Koenker 1997). Its solution is the exact
    minimum when no far pair changes sign. Otherwise the near set grows, up to
    all pairs. When the minimum is not unique, as with group indicators, the
    minimizer with the smallest residual sum of squares is returned.
    """
    n = len(y)
    size, beta = near_pairs * n, _jaeckel_pilot(X, y)
    while size < n * (n - 1) // 2:
        beta, certified = _near_pairs_fit(X, y, beta, size)
        if certified:
            return beta
        size *= 4
    i, j = np.triu_indices(n, 1)
    Z, d = X[i] - X[j], y[i] - y[j]
    keep = Z.any(axis=1)  # pairs with equal x only add a constant
    return _least_squares_minimizer(X, y, Z[keep], d[keep], np.zeros(X.shape[1]))


def _jaeckel_pilot(X, y):
    """Approximate minimizer of Jaeckel's dispersion by BFGS, as Rfit's ``jaeckel``."""
    scale = np.max(np.abs(y)) or 1.0
    sides = 2.0 * np.arange(1, len(y) + 1) - len(y) - 1  # Wilcoxon scores of sorted residuals

    def dispersion(beta):
        residuals = y / scale - X @ beta
        order = np.argsort(residuals)
        return sides @ residuals[order], -(sides @ X[order])

    start = np.linalg.lstsq(X, y / scale, rcond=None)[0]
    pilot = optimize.minimize(dispersion, start, jac=True, method="BFGS").x
    return scale * (pilot if np.isfinite(pilot).all() else start)


def _near_pairs_fit(X, y, beta, size):
    """L1 fit on the ``size`` closest residual pairs at ``beta``, and whether it is exact."""
    n = len(y)
    residuals = y - X @ beta
    order = np.argsort(residuals)
    ordered = residuals[order]
    first, last = _pairwise_rows(n)
    with np.errstate(over="ignore"):
        delta = _kth_smallest(lambda i, j: ordered[j] - ordered[i], first, last, size)
    stop = np.searchsorted(ordered, ordered + delta, side="right")  # near: i < j < stop[i]
    counts = stop - np.arange(1, n + 1)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    i = np.repeat(np.arange(n), counts)
    i, j = order[i], order[i + 1 + np.arange(counts.sum()) - starts]
    Z, d = X[i] - X[j], y[i] - y[j]
    # A far pair adds sign(r_i - r_j) (d_ij - z_ij' b). Over all pairs these signs
    # weight row i by 2 rank(r_i) - n - 1; subtract the near pairs.
    sides = 2 * stats.rankdata(residuals) - n - 1
    linear = np.sign(residuals[i] - residuals[j]) @ Z - sides @ X
    solution = _least_squares_minimizer(X, y, Z, d, linear)
    far = stop < n
    gap = np.min(ordered[stop[far]] - ordered[far]) if far.any() else np.inf
    # No pair moves by more than the range of the fitted change.
    if solution is None:
        return beta, False
    return solution, bool(np.ptp(X @ (solution - beta)) < gap)


def _least_squares_minimizer(X, y, Z, d, linear):
    """Minimizer of ``sum |d - Z b| + linear' b`` with the smallest ``||y - X b||``.

    For a dual solution ``lam``, ``b`` is a minimizer exactly when ``d_k - z_k' b``
    is zero where ``|lam_k| < 1`` and has the sign of ``lam_k`` elsewhere. Least
    squares over that polytope is unique and does not depend on how the columns
    of ``X`` code the design. Returns None if the objective is unbounded below.
    """
    solved = _l1_solve(Z, d, linear)
    if solved is None:
        return None
    beta, lam = solved
    tight = np.abs(lam) < 1 - 1e-6
    _, values, vectors = np.linalg.svd(Z[tight].reshape(-1, Z.shape[1]))
    rank = np.count_nonzero(values > values.max(initial=0) * max(Z.shape) * np.finfo(float).eps)
    null = vectors[rank:].T  # directions along the set of minimizers
    if not null.shape[1]:
        return beta
    side = np.sign(lam[~tight])
    bound = Z[~tight]
    G = -side[:, None] * (bound @ null)
    h = -np.maximum(side * (d[~tight] - bound @ beta), 0)
    moving = np.linalg.norm(G, axis=1) > 1e-9 * np.linalg.norm(bound, axis=1)
    residuals = y - X @ beta
    step = null @ _least_squares_with_inequalities(X @ null, residuals, G[moving], h[moving])
    if np.max(np.abs(X @ step)) <= 1e-12 * np.max(np.abs(residuals)):
        return beta  # a rounding-level step: keep tied residuals exactly tied
    candidate = beta + step

    def objective(b):
        return np.abs(d - Z @ b).sum() + linear @ b

    tolerance = 1e-10 * (np.abs(d - Z @ beta).sum() + np.abs(linear) @ np.abs(beta))
    if not objective(candidate) <= objective(beta) + tolerance:  # pragma: no cover
        return beta  # a dual value misread at the solver's tolerance
    return candidate


def _least_squares_with_inequalities(A, e, G, h):
    """``t`` minimizing ``||A t - e||`` subject to ``G t >= h``, with ``t = 0`` feasible.

    Least distance programming solved by nonnegative least squares (Lawson &
    Hanson 1974, chapter 23). ``A`` must have full column rank.
    """
    Q, R = np.linalg.qr(A)
    center = Q.T @ e
    G = np.vstack([G, np.zeros(len(R))])  # a constraint that always holds
    h = np.r_[h, -1.0]
    scaled = linalg.solve_triangular(R, G.T, trans=1).T  # G R^-1
    E = np.vstack([scaled.T, h - scaled @ center])
    weights, _ = optimize.nnls(E, np.r_[np.zeros(len(R)), 1.0])
    residual = E @ weights
    residual[-1] -= 1
    return linalg.solve_triangular(R, center - residual[:-1] / residual[-1])


def _l1_fit(Z, d):
    """Coefficients ``b`` minimizing ``sum |d - Z b|`` (least absolute deviations)."""
    return _l1_solve(Z, d, np.zeros(Z.shape[1]))[0]


def _l1_solve(Z, d, linear):
    """Minimizer of ``sum |d - Z b| + linear' b`` and its dual, or None if unbounded.

    Solved through the dual linear program ``max d' lam`` subject to
    ``Z' lam = linear`` and ``-1 <= lam <= 1``, whose equality multipliers are ``b``.
    """
    x_scale = np.max(np.abs(Z), axis=0, initial=0)
    x_scale[x_scale == 0] = 1.0
    y_scale = np.max(np.abs(d), initial=0) or 1.0
    fit = optimize.linprog(
        -d / y_scale, A_eq=(Z / x_scale).T, b_eq=linear / x_scale, bounds=(-1, 1)
    )
    if fit.status == 2:  # infeasible dual: unbounded below
        return None
    if fit.status != 0:  # pragma: no cover (bounded and feasible otherwise)
        raise RuntimeError(f"linear programming failed: {fit.message}")
    return -fit.eqlin.marginals * (y_scale / x_scale), fit.x


def _dispersion(residuals):
    """Jaeckel's dispersion with Wilcoxon scores standardized as Rfit."""
    n = len(residuals)
    scores = math.sqrt(12) * (np.arange(1, n + 1) - (n + 1) / 2) / math.sqrt(n * (n - 1))
    return float(scores @ np.sort(residuals))


def _tau(residuals, p, delta=0.8):
    """Koul-Sievers-McKean estimate of tau for Wilcoxon scores, with Rfit's corrections.

    ``1 / ((a(n) - a(1)) f2)`` with ``a`` the standardized scores (so the range
    tends to ``sqrt(12)``), ``f2 = H(t) / (2 t)`` the density of ``e_i - e_j`` at
    zero, ``H`` the empirical distribution of ``|e_i - e_j|`` over pairs and the
    window ``t`` its ``delta`` quantile (exact, where Rfit solves ``H(t) = delta``
    to an absolute tolerance of 1.2e-4) divided by ``sqrt(n)``; then scaled by
    ``sqrt(n / (n - p))`` and a correction for the residuals beyond twice their MAD.
    """
    n = len(residuals)
    window, share = _tau_window(residuals, delta)
    if window == 0 or share == 0:
        return np.nan
    tau = 2 * window / (math.sqrt(12 * (n - 1) / n) * share)
    inside = np.mean(np.abs(residuals - np.median(residuals)) < 2 * mad(residuals))
    inside = max(inside, 1e-6)
    return float(math.sqrt(n / (n - p)) * tau * (1 + p / n * (1 - inside) / inside))


def _tau_window(residuals, delta):
    """The Koul window and the share of pair distances at or below it."""
    n = len(residuals)
    pairs = n * (n - 1) // 2
    # The delta quantile of H: the smallest difference with H >= delta, the
    # root of H(t) = delta that Rfit's default routine approximates numerically.
    rank = math.ceil(delta * pairs - 1e-9)
    if n < _TAU_IMPLICIT_MIN_SIZE:
        distances = pdist(residuals[:, None], metric="cityblock")
        distances.partition(rank - 1)
        window = distances[rank - 1] / math.sqrt(n)
        return window, np.mean(distances <= window)

    ordered = np.sort(residuals)
    first, last = _pairwise_rows(n)

    def distance(i, j):
        return ordered[j] - ordered[i]

    # Distances beyond float64's range compare as +inf, as they do in pdist.
    with np.errstate(over="ignore"):
        window = _kth_smallest(distance, first, last, rank) / math.sqrt(n)
        through = _count(distance, np.arange(first.size), first, last, window, strict=False)
    return window, through.sum() / pairs


def _tau_s(residuals, rank, confidence=0.95):
    """``tau_s`` from the distribution-free interval for the median, as Rfit's ``taustar``."""
    n = len(residuals)
    z = stats.norm.ppf((1 + confidence) / 2)
    c = max(math.floor(0.5 * n - 0.5 * math.sqrt(n) * z - 0.5), 0)
    ordered = np.sort(residuals)
    width = ordered[n - c - 1] - ordered[c]
    # Rfit uses n - rank - 1. Retain its correction, bounded by one
    # effective degree of freedom when the residual degrees of freedom is one.
    return float(math.sqrt(n / max(n - rank - 1, 1)) * math.sqrt(n) * width / (2 * z))


class RankTestResult(NamedTuple):
    """Result of :func:`rank_sum_test` and :func:`signed_rank_test`."""

    statistic: float
    pvalue: float


def rank_sum_test(x, y, *, alternative="two-sided"):
    """Exact Wilcoxon-Mann-Whitney test, conditional on ties.

    The statistic is ``U = W - m (m + 1) / 2``, with ``W`` the sum of the
    midranks of ``x`` in the pooled sample of sizes ``m`` and ``n``. Under the
    hypothesis that both samples come from the same distribution, every
    assignment of the observed midranks to the samples is equally likely; the
    p-value is computed from that exact distribution, by the shift algorithm
    of Streitberg & Röhmel (1986) on doubled (integer) midranks. With ties this
    differs from the distribution without ties used by many programs (scipy's
    ``mannwhitneyu(method="exact")`` among them). The two-sided p-value is
    ``P(|W - E W| >= |w - E W|)``, as R's coin and exactRankTests. The cost
    grows as ``m (m + n)^3``: a few hundred observations take seconds.

    Parameters
    ----------
    x, y : array-like of shape (m,) and (n,)
        Samples; ties are allowed.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        "greater": ``x`` tends to be larger than ``y``.

    Returns
    -------
    RankTestResult
        ``statistic`` (``U``) and ``pvalue``.

    References
    ----------
    Streitberg, B. and Röhmel, J. (1986). Exact distributions for permutation
    and rank tests: an introduction to some recently published algorithms.
    *Statistical Software Newsletter*, 12(1), 10-17.

    Hothorn, T., Hornik, K., van de Wiel, M. A. and Zeileis, A. (2006). A Lego
    system for conditional inference. *The American Statistician*, 60(3),
    257-263.

    Examples
    --------
    >>> from robustmodels.rank import rank_sum_test
    >>> result = rank_sum_test([1, 2, 2, 3, 3, 3, 5], [2, 3, 4, 4, 5, 6])
    >>> result.statistic, round(result.pvalue, 4)
    (10.0, 0.1166)
    """
    check_option("alternative", alternative, _ALTERNATIVES)
    x, y = as_1d(x), as_1d(y)
    m, total = len(x), len(x) + len(y)
    if not m or not len(y):
        raise ValueError("both samples need at least one observation")
    scores = np.rint(2 * stats.rankdata(np.r_[x, y])).astype(np.int64)
    k = min(m, len(y))  # subsets of the smaller sample need less memory
    limit = int(np.sort(scores)[-k:].sum())
    counts = np.zeros((k + 1, limit + 1))  # counts[j, s]: j-subsets with score sum s
    counts[0, 0] = 1
    for i, score in enumerate(scores):
        top = min(i + 1, k)
        counts[1 : top + 1, score:] += counts[:top, : limit + 1 - score]
    probabilities = counts[k] / counts[k].sum()
    sums = np.arange(limit + 1) if k == m else int(scores.sum()) - np.arange(limit + 1)
    observed = int(scores[:m].sum())
    centred = total * sums - m * int(scores.sum())  # total * (W - E W), doubled ranks
    pvalue = _pvalue(centred, probabilities, total * observed - m * int(scores.sum()), alternative)
    return RankTestResult(observed / 2 - m * (m + 1) / 2, pvalue)


def signed_rank_test(x, y=None, *, mu=0.0, zero_method="wilcox", alternative="two-sided"):
    """Exact Wilcoxon signed-rank test, conditional on ties and zeros.

    The statistic ``V`` is the sum of the midranks of ``|d|`` over the
    positive differences ``d = (x - y) - mu`` (or ``x - mu``). Under the hypothesis
    that ``d`` is symmetric about zero, each sign is equally likely given the
    observed midranks; the p-value is computed from that exact distribution
    on doubled (integer) midranks. Zero differences are dropped before ranking
    (``"wilcox"``, as R's ``wilcox.test``) or ranked with the others and given
    no sign (``"pratt"``, as coin's default). The distribution is symmetric,
    so the two-sided p-value doubles the smaller tail.

    Parameters
    ----------
    x : array-like of shape (n,)
        Sample, or first measurements of pairs.
    y : array-like of shape (n,), optional
        Second measurements of the pairs.
    mu : float, default 0.0
        Center under the hypothesis, subtracted from ``x - y``.
    zero_method : {"wilcox", "pratt"}, default "wilcox"
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        "greater": the differences tend to be positive.

    Returns
    -------
    RankTestResult
        ``statistic`` (``V``) and ``pvalue``; 1 when every difference is zero.

    Raises
    ------
    ValueError
        If ``mu`` is not a finite scalar or a difference overflows float64.

    References
    ----------
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    *Biometrics Bulletin*, 1(6), 80-83.

    Pratt, J. W. (1959). Remarks on zeros and ties in the Wilcoxon signed rank
    procedures. *Journal of the American Statistical Association*, 54(287),
    655-667.

    Examples
    --------
    >>> from robustmodels.rank import signed_rank_test
    >>> result = signed_rank_test([0.5, -1, 1, 2, 2, -0.5, 3, 1, 0, 1.5])
    >>> result.statistic, round(result.pvalue, 4)
    (39.5, 0.0508)
    """
    check_option("alternative", alternative, _ALTERNATIVES)
    check_option("zero_method", zero_method, ("wilcox", "pratt"))
    if isinstance(mu, np.generic):
        mu = mu.item()  # a float32 compared with the float64 limit would overflow
    limit = float(np.finfo(float).max)
    if isinstance(mu, bool) or not isinstance(mu, Real) or not -limit <= mu <= limit:
        raise ValueError("mu must be a finite scalar")
    x = as_1d(x)
    if y is not None:
        y = as_1d(y)
        if len(y) != len(x):
            raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    d = _signed_differences(x, y, float(mu))
    if zero_method == "wilcox":
        d = d[d != 0]
    scores = np.rint(2 * stats.rankdata(np.abs(d))).astype(np.int64)
    scores = scores[d != 0]
    positive = scores[d[d != 0] > 0]
    probabilities = np.zeros(int(scores.sum()) + 1)
    probabilities[0] = 1
    for score in scores:  # each sign with probability 1/2; halving is exact
        probabilities[score:] += probabilities[: len(probabilities) - score].copy()
        probabilities *= 0.5
    observed, total = int(positive.sum()), int(scores.sum())
    centred = 2 * np.arange(total + 1) - total  # 2 (V - E V), doubled ranks
    pvalue = _pvalue(centred, probabilities, 2 * observed - total, alternative)
    return RankTestResult(observed / 2, pvalue)


def _signed_differences(x, y, mu):
    with np.errstate(over="raise"):
        try:
            if y is None:
                return x - mu
            if mu == 0:
                return x - y
            pairs = (_shifted_difference(a, b, mu) for a, b in zip(x, y, strict=True))
            return np.fromiter(pairs, dtype=float, count=len(x))
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("signed differences overflow float64. Rescale x, y and mu") from exc


def _shifted_difference(a, b, mu):
    terms = (float(a), -float(b), -mu)
    try:
        return math.fsum(terms)
    except OverflowError:
        # fsum may overflow a partial sum even if the final result fits.
        return float(sum(map(Fraction, terms)))


def _pvalue(centred, probabilities, observed, alternative):
    """Exact p-value of the centred statistic ``observed`` (integers, so ties compare exactly)."""
    if alternative == "greater":
        tail = centred >= observed
    elif alternative == "less":
        tail = centred <= observed
    else:
        tail = np.abs(centred) >= abs(observed)
    return float(min(1.0, probabilities[tail].sum()))
