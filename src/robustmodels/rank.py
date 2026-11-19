"""Rank-based regression and exact rank tests that condition on ties.

:class:`RankRegressor` is a scikit-learn regressor; ``fit(X, y)`` sets
``coef_``, ``intercept_``, the scale parameters of its inference and
coefficient tables, and ``predict(X)`` returns ``X @ coef_ + intercept_``.
:func:`rank_sum_test` and :func:`signed_rank_test` compute exact p-values from
the permutation distribution of midrank scores given the observed ties, which
remains valid with ties and zeros.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy import linalg, optimize, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_1d, check_option
from .regression import _coefficient_table
from .scale import mad

__all__ = [
    "RankRegressor",
    "RankTestResult",
    "drop_test",
    "rank_anova",
    "rank_sum_test",
    "signed_rank_test",
]

_ALTERNATIVES = ("two-sided", "less", "greater")


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
        self.coef_ = _wilcoxon_slopes(centred, y)
        residuals = y - X @ self.coef_
        self.intercept_ = float(np.median(residuals))
        residuals = residuals - self.intercept_
        self.scale_ = mad(residuals)
        self.dispersion_ = _dispersion(residuals)
        self._null_dispersion, self._n_samples = _dispersion(y), n
        self.tau_ = _tau(residuals, p)
        self.tau_s_ = _tau_s(residuals, p + 1)
        design = np.column_stack([np.ones(n), X])
        cov = self.tau_**2 * linalg.pinvh(design.T @ design)
        cov[0, 0] += (self.tau_s_**2 - self.tau_**2) / n
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
    analogue of the F test of nested linear models and keeps its level when
    the errors are not normal.

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
        same = np.isclose(reduced._null_dispersion, full._null_dispersion)
        if reduced._n_samples != n or not same:
            raise ValueError("full and reduced must be fitted to the same response")
        dropped, q = reduced.dispersion_, p - reduced.n_features_in_
        if q < 1:
            raise ValueError("the reduced model must have fewer columns than the full model")
    reduction = dropped - full.dispersion_
    if reduction < -1e-8 * max(dropped, 1e-300):
        raise ValueError("the full model has a larger dispersion: the models are not nested")
    statistic = max(reduction, 0.0) / q / (full.tau_ / 2)
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


def _wilcoxon_slopes(X, y):
    """Slopes minimizing the sum of ``|(y_i - y_j) - (x_i - x_j)' b|`` over pairs ``i < j``.

    Solved through the dual linear program ``max d' lam`` subject to
    ``Z' lam = 0`` and ``-1 <= lam <= 1``, whose equality multipliers are the
    slopes; pairs with equal ``x`` only add a constant and are left out.
    """
    i, j = np.triu_indices(len(y), 1)
    Z, d = X[i] - X[j], y[i] - y[j]
    keep = Z.any(axis=1)
    Z, d = Z[keep], d[keep]
    fit = optimize.linprog(-d, A_eq=Z.T, b_eq=np.zeros(X.shape[1]), bounds=(-1, 1))
    if fit.status != 0:  # pragma: no cover (a bounded, feasible problem: lam = 0)
        raise RuntimeError(f"linear programming failed: {fit.message}")
    return -fit.eqlin.marginals


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
    differences = np.sort(np.abs(_pair_differences(residuals)))
    # The delta quantile of H: the smallest difference with H >= delta, the
    # root of H(t) = delta that Rfit's default routine approximates numerically.
    k = math.ceil(delta * len(differences) - 1e-9) - 1
    window = differences[k] / math.sqrt(n)
    share = np.searchsorted(differences, window, "right") / len(differences)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = 2 * window / (math.sqrt(12 * (n - 1) / n) * share)
        inside = np.mean(np.abs(residuals - np.median(residuals)) < 2 * mad(residuals))
    inside = max(inside, 1e-6)
    return float(math.sqrt(n / (n - p)) * tau * (1 + p / n * (1 - inside) / inside))


def _tau_s(residuals, rank, confidence=0.95):
    """``tau_s`` from the distribution-free interval for the median, as Rfit's ``taustar``."""
    n = len(residuals)
    z = stats.norm.ppf((1 + confidence) / 2)
    c = max(math.floor(0.5 * n - 0.5 * math.sqrt(n) * z - 0.5), 0)
    ordered = np.sort(residuals)
    width = ordered[n - c - 1] - ordered[c]
    return float(math.sqrt(n / (n - rank - 1)) * math.sqrt(n) * width / (2 * z))


def _pair_differences(x):
    i, j = np.triu_indices(len(x), 1)
    return x[i] - x[j]


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
    positive differences ``d = x - y`` (or ``x - mu``). Under the hypothesis
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
    d = as_1d(x) - mu
    if y is not None:
        y = as_1d(y)
        if len(y) != len(d):
            raise ValueError(f"x and y must have the same length, got {len(d)} and {len(y)}")
        d = d - y
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


def _pvalue(centred, probabilities, observed, alternative):
    """Exact p-value of the centred statistic ``observed`` (integers, so ties compare exactly)."""
    if alternative == "greater":
        tail = centred >= observed
    elif alternative == "less":
        tail = centred <= observed
    else:
        tail = np.abs(centred) >= abs(observed)
    return float(min(1.0, probabilities[tail].sum()))
