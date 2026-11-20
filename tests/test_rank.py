"""Exact enumeration, optimality of the rank fit, calibration and frozen R values.

Reference values: tests/data/r_rank.json (Rfit, exactRankTests and coin; its
"meta" entry records the versions). Rfit minimizes the dispersion by
quasi-Newton steps and solves H(t) = 0.8 for tau to an absolute tolerance of
1.2e-4, so its coefficients and tau differ slightly from the exact values.
"""

import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels import rank
from robustmodels.rank import (
    RankRegressor,
    _dispersion,
    _least_squares_minimizer,
    _tau,
    _tau_s,
    _tau_window,
    _wilcoxon_slopes,
    drop_test,
    rank_anova,
    rank_sum_test,
    signed_rank_test,
)

R = json.loads((Path(__file__).parent / "data" / "r_rank.json").read_text())
ALTERNATIVES = ["two-sided", "less", "greater"]


@parametrize_with_checks([RankRegressor()])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def brute_rank_sum(x, y, alternative):
    """p-value by enumerating every assignment of the pooled midranks."""
    ranks = stats.rankdata(np.r_[x, y])
    m, total = len(x), len(x) + len(y)
    sums = np.array([ranks[list(c)].sum() for c in itertools.combinations(range(total), m)])
    observed, center = ranks[:m].sum(), m * ranks.mean()
    if alternative == "greater":
        return np.mean(sums >= observed - 1e-9)
    if alternative == "less":
        return np.mean(sums <= observed + 1e-9)
    return np.mean(np.abs(sums - center) >= abs(observed - center) - 1e-9)


def brute_signed_rank(d, alternative, zero_method):
    """p-value by enumerating every sign pattern of the nonzero differences."""
    d = np.asarray(d, float)
    if zero_method == "wilcox":
        d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))[d != 0]
    signs = np.array(list(itertools.product([0, 1], repeat=len(ranks))))
    sums = signs @ ranks
    observed, center = ranks[d[d != 0] > 0].sum(), ranks.sum() / 2
    if alternative == "greater":
        return np.mean(sums >= observed - 1e-9)
    if alternative == "less":
        return np.mean(sums <= observed + 1e-9)
    return np.mean(np.abs(sums - center) >= abs(observed - center) - 1e-9)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_rank_sum_by_enumeration_with_ties(alternative):
    rng = np.random.default_rng(0)
    for _ in range(40):
        m, n = rng.integers(1, 8, size=2)
        x, y = rng.integers(0, 4, size=m) / 2, rng.integers(0, 5, size=n) / 2
        expected = brute_rank_sum(x, y, alternative)
        assert rank_sum_test(x, y, alternative=alternative).pvalue == pytest.approx(expected)
        assert rank_sum_test(y, x, alternative=alternative).pvalue == pytest.approx(
            brute_rank_sum(y, x, alternative)
        )


@pytest.mark.parametrize("zero_method", ["wilcox", "pratt"])
@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_signed_rank_by_enumeration_with_ties_and_zeros(alternative, zero_method):
    rng = np.random.default_rng(1)
    for _ in range(40):
        d = rng.integers(-3, 4, size=rng.integers(1, 13)) / 2
        expected = brute_signed_rank(d, alternative, zero_method)
        result = signed_rank_test(d, alternative=alternative, zero_method=zero_method)
        assert result.pvalue == pytest.approx(expected)


def test_rank_tests_agree_with_scipy_without_ties():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=9), rng.normal(size=11) + 0.5
    for alternative in ALTERNATIVES:
        ours = rank_sum_test(x, y, alternative=alternative)
        theirs = stats.mannwhitneyu(x, y, alternative=alternative, method="exact")
        assert ours.statistic == theirs.statistic
        assert ours.pvalue == pytest.approx(theirs.pvalue, rel=1e-12)
        ours = signed_rank_test(x, y[:9], alternative=alternative)
        theirs = stats.wilcoxon(x, y[:9], alternative=alternative, method="exact")
        assert ours.pvalue == pytest.approx(theirs.pvalue, rel=1e-12)


def test_signed_rank_paired_shift_and_degenerate_cases():
    x, y = np.array([3.0, 5, 2, 8, 7]), np.array([1.0, 4, 2, 5, 9])
    assert signed_rank_test(x, y) == signed_rank_test(x - y)
    assert signed_rank_test(x, mu=2.0) == signed_rank_test(x - 2.0)
    assert signed_rank_test([0.0, 0.0]) == (0.0, 1.0)
    with pytest.raises(ValueError, match="same length"):
        signed_rank_test(x, y[:4])
    with pytest.raises(ValueError, match="at least one"):
        rank_sum_test([], [1.0])
    with pytest.raises(ValueError, match="alternative"):
        rank_sum_test([1.0], [2.0], alternative="two.sided")


def test_signed_rank_preserves_small_differences_after_large_cancellation():
    for x, y, mu in [(1e16, 1e16, 1.0), (1e16, 1.0, 1e16)]:
        result = signed_rank_test(np.full(8, x), np.full(8, y), mu=mu, alternative="less")
        assert result == (0.0, 1 / 256)


def test_signed_rank_keeps_finite_differences_after_intermediate_overflow():
    result = signed_rank_test([1e308, 0], [-1e308, -1e308], mu=1e308, zero_method="pratt")
    assert result == signed_rank_test([1e308, 0], zero_method="pratt") == (2.0, 1.0)


@pytest.mark.parametrize("y, mu", [(None, -1e308), ([-1e308], 0), ([-1e308], -1e308)])
def test_signed_rank_rejects_unrepresentable_differences(y, mu):
    with pytest.raises(ValueError, match="differences overflow"):
        signed_rank_test([1e308], y, mu=mu)


@pytest.mark.parametrize(
    "mu", [np.nan, np.inf, [0.0], True, np.bool_(False), pytest.param(10**400, id="overflow")]
)
def test_signed_rank_requires_finite_scalar_shift(mu):
    with pytest.raises(ValueError, match="mu must be a finite scalar"):
        signed_rank_test([1.0], mu=mu)


@pytest.mark.parametrize("mu", [np.float32(0.5), np.int8(1), np.longdouble(0.5)])
def test_signed_rank_accepts_numpy_scalar_shift_without_warnings(mu):
    x = [1.0, 2.0, 3.0, -1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert signed_rank_test(x, mu=mu) == signed_rank_test(x, mu=float(mu))


@pytest.mark.parametrize("name", [k for k in R["tests"] if k.startswith("two")])
def test_rank_sum_r_reference(name):
    r = R["tests"][name]
    results = [rank_sum_test(r["x"], r["y"], alternative=a) for a in ALTERNATIVES]
    assert results[0].statistic == r["W"]
    np.testing.assert_allclose([res.pvalue for res in results], r["p"], rtol=1e-10)
    assert results[0].pvalue == pytest.approx(r["coin_two_sided"], rel=1e-10)


@pytest.mark.parametrize("name", [k for k in R["tests"] if k.startswith("one")])
def test_signed_rank_r_reference(name):
    r = R["tests"][name]
    results = [signed_rank_test(r["d"], alternative=a) for a in ALTERNATIVES]
    assert results[0].statistic == r["V"]
    np.testing.assert_allclose([res.pvalue for res in results], r["p"], rtol=1e-10)
    pratt = signed_rank_test(r["d"], zero_method="pratt").pvalue
    assert pratt == pytest.approx(r["coin_pratt_two_sided"], rel=1e-10)


def pairwise_loss(X, y, coef):
    e = y - X @ coef
    i, j = np.triu_indices(len(y), 1)
    return np.abs(e[i] - e[j]).sum()


def test_dispersion_equals_scaled_sum_of_pairwise_differences():
    e = np.random.default_rng(3).normal(size=17)
    i, j = np.triu_indices(17, 1)
    expected = math.sqrt(3 / (17 * 16)) * np.abs(e[i] - e[j]).sum()
    assert _dispersion(e) == pytest.approx(expected, rel=1e-13)
    # The standardized Wilcoxon scores have mean 0 and sum of squares n + 1.
    scores = math.sqrt(12) * (np.arange(1, 18) - 9) / math.sqrt(17 * 16)
    assert scores.sum() == pytest.approx(0, abs=1e-12)
    assert (scores**2).sum() == pytest.approx(18)


def test_simple_regression_is_the_weighted_median_of_pairwise_slopes():
    rng = np.random.default_rng(4)
    for _ in range(10):
        x = rng.normal(size=25)
        y = 1 + 2 * x + rng.standard_t(2, size=25)
        i, j = np.triu_indices(25, 1)
        slopes, weights = (y[i] - y[j]) / (x[i] - x[j]), np.abs(x[i] - x[j])
        order = np.argsort(slopes)
        cumulative = np.cumsum(weights[order])
        median = slopes[order][np.searchsorted(cumulative, cumulative[-1] / 2)]
        coef = RankRegressor().fit(x[:, None], y).coef_[0]
        loss = [pairwise_loss(x[:, None], y, [b]) for b in (coef, median)]
        assert loss[0] == pytest.approx(loss[1], rel=1e-12)


def test_fit_minimizes_the_dispersion():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.standard_t(3, size=40)
    fit = RankRegressor().fit(X, y)
    best = pairwise_loss(X, y, fit.coef_)
    for step in rng.normal(scale=1e-3, size=(200, 3)):
        assert pairwise_loss(X, y, fit.coef_ + step) >= best - 1e-9
    assert fit.intercept_ == pytest.approx(np.median(y - X @ fit.coef_))


@pytest.mark.parametrize("errors", ["cauchy", "ties"])
@pytest.mark.parametrize("pilot", ["bfgs", "zero"])
def test_near_pair_fit_matches_the_linear_program_on_all_pairs(errors, pilot, monkeypatch):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(100, 3))
    noise = rng.standard_cauchy(100) if errors == "cauchy" else rng.integers(-2, 3, 100)
    y = X @ [1.0, -2.0, 0.5] + noise
    if errors == "ties":
        X, y = np.round(X), np.round(y)
    X -= X.mean(axis=0)
    i, j = np.triu_indices(len(y), 1)
    exact = _least_squares_minimizer(X, y, X[i] - X[j], y[i] - y[j], np.zeros(3))
    sizes = []
    near_pairs_fit = rank._near_pairs_fit

    def recorded(X, y, beta, size):
        sizes.append(size)
        return near_pairs_fit(X, y, beta, size)

    monkeypatch.setattr(rank, "_near_pairs_fit", recorded)
    if pilot == "zero":  # a poor pilot fails the first certificate, so the near set grows
        monkeypatch.setattr(rank, "_jaeckel_pilot", lambda X, y: np.zeros(X.shape[1]))
    fit = _wilcoxon_slopes(X, y)
    assert pilot == "bfgs" or len(sizes) > 1
    assert pairwise_loss(X, y, fit) == pytest.approx(pairwise_loss(X, y, exact), rel=1e-12)
    np.testing.assert_allclose(fit, exact, rtol=1e-9, atol=1e-12)


def test_equivariance():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 2))
    y = 1 + X @ [1.0, 2.0] + rng.standard_t(3, size=30)
    fit = RankRegressor().fit(X, y)
    A = np.array([[2.0, 1.0], [0.5, -1.0]])
    other = RankRegressor().fit(X @ A, 3 * y + X @ [1.0, -1.0] + 4)
    np.testing.assert_allclose(other.coef_, np.linalg.solve(A, 3 * fit.coef_ + [1.0, -1.0]))
    assert other.intercept_ == pytest.approx(3 * fit.intercept_ + 4)
    inverse = np.linalg.inv(A)
    expected = 9 * inverse @ fit.cov_params_[1:, 1:] @ inverse.T
    np.testing.assert_allclose(other.cov_params_[1:, 1:], expected, rtol=1e-8)
    assert other.tau_ == pytest.approx(3 * fit.tau_, rel=1e-10)
    assert other.tau_s_ == pytest.approx(3 * fit.tau_s_, rel=1e-10)


@pytest.mark.parametrize("min_size", [0, 2048])
def test_tau_includes_pairs_at_the_window_boundary(monkeypatch, min_size):
    monkeypatch.setattr(rank, "_TAU_IMPLICIT_MIN_SIZE", min_size)
    # Four of the six differences (0, 2, 2, 2, 4, 4) are at most 4 / sqrt(4).
    assert _tau(np.array([0.0, 0.0, 2.0, 4.0]), 0) == pytest.approx(2.0)


@pytest.mark.parametrize("delta", [0.5, 0.8])
def test_tau_window_agrees_with_explicit_distances_for_large_tied_sample(delta):
    residuals = np.random.default_rng(103).integers(-20, 21, size=2048).astype(float)
    i, j = np.triu_indices(len(residuals), 1)
    distances = np.abs(residuals[i] - residuals[j])
    k = math.ceil(delta * len(distances) - 1e-9) - 1
    window = np.sort(distances)[k] / math.sqrt(len(residuals))
    share = np.mean(distances <= window)
    assert _tau_window(residuals, delta) == (window, share)


def test_tau_implicit_window_preserves_overflowing_distances(monkeypatch):
    residuals = np.array([-1e308, -1e308, 0.0, 1e308])
    expected = [_tau_window(residuals, delta) for delta in (0.5, 0.8)]
    monkeypatch.setattr(rank, "_TAU_IMPLICIT_MIN_SIZE", 0)
    assert expected == [(5e307, 1 / 6), (np.inf, 1.0)]
    assert [_tau_window(residuals, delta) for delta in (0.5, 0.8)] == expected


def test_tau_estimates_the_wilcoxon_scale_at_the_normal_model():
    # tau = 1 / (sqrt(12) integral f^2) = sqrt(pi / 3) for standard normal errors,
    # and tau_s = 1 / (2 f(0)) = sqrt(pi / 2).
    # tau_s uses about sqrt(n) central order statistics, so it varies more;
    # the mean of 30 samples has a relative standard error near 1%.
    rng = np.random.default_rng(7)
    assert _tau(rng.normal(size=4000), 0) == pytest.approx(math.sqrt(math.pi / 3), rel=0.03)
    mean = np.mean([_tau_s(rng.normal(size=20000), 1) for _ in range(30)])
    assert mean == pytest.approx(math.sqrt(math.pi / 2), rel=0.04)


def test_drop_test_and_summary():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] + rng.standard_t(3, size=50)
    full, reduced = RankRegressor().fit(X, y), RankRegressor().fit(X[:, :1], y)
    test = drop_test(full, reduced)
    expected = (reduced.dispersion_ - full.dispersion_) / 2 / (full.tau_ / 2)
    assert test.statistic == pytest.approx(expected)
    assert test.df == (2, 46)
    assert test.pvalue == pytest.approx(stats.f.sf(expected, 2, 46))
    text = full.summary(["a", "b", "c"])
    assert "(Intercept)" in text and "Drop in dispersion test" in text
    with pytest.raises(ValueError, match="fewer columns"):
        drop_test(reduced, full)
    with pytest.raises(ValueError, match="same response"):
        drop_test(full, RankRegressor().fit(X[:, :1], 2 * y))
    with pytest.raises(ValueError, match="not nested"):
        drop_test(RankRegressor().fit(X[:, 1:], y), RankRegressor().fit(X[:, :1], y))


def test_input_checks():
    with pytest.raises(ValueError, match="more observations"):
        RankRegressor().fit(np.ones((2, 1)), [1.0, 2.0])
    with pytest.raises(ValueError, match="singular"):
        RankRegressor().fit(np.c_[np.arange(5.0), 2 * np.arange(5.0)], np.arange(5.0))


def test_rank_fit_with_one_residual_degree_of_freedom():
    X = np.array([[-1.0], [0.0], [1.0]])
    fit = RankRegressor().fit(X, [1.0, 2.0, 4.0])
    assert fit.coef_[0] == pytest.approx(1.5)
    assert np.isfinite(fit.std_errors_).all()
    assert drop_test(fit).df == (1, 1)


@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_rank_fit_is_equivariant_in_extreme_response_units(scale):
    X = np.arange(5, dtype=float)[:, None]
    y = np.array([2, 3, 5, 7, 9.5])
    baseline = RankRegressor().fit(X, y)
    fit = RankRegressor().fit(X, scale * y)
    np.testing.assert_allclose(fit.predict(X) / scale, baseline.predict(X))
    np.testing.assert_allclose(fit.std_errors_ / scale, baseline.std_errors_)


def test_exact_and_degenerate_rank_fits_have_no_inference():
    X = np.arange(20, dtype=float)[:, None]
    for y in (1 + 2 * X[:, 0], np.r_[np.zeros(19), 1]):
        fit = RankRegressor().fit(X, y)
        assert np.isnan(fit.std_errors_).all()
        assert np.isnan(drop_test(fit).pvalue)


def test_drop_test_rejects_response_reordering_and_nonnested_designs():
    rng = np.random.default_rng(64)
    X = rng.normal(size=(24, 3))
    y = X[:, 0] + rng.normal(size=24)
    full = RankRegressor().fit(X[:, :2], y)
    with pytest.raises(ValueError, match="same response"):
        drop_test(full, RankRegressor().fit(X[:, :1], y[::-1]))
    with pytest.raises(ValueError, match="not nested"):
        drop_test(full, RankRegressor().fit(X[:, 2:], y))
    reduced = RankRegressor().fit((X[:, :1] + X[:, 1:2]) * 3 + 8, y)
    assert np.isfinite(drop_test(full, reduced).pvalue)


def test_interval_coverage_and_test_size():
    """Monte Carlo: 95% slope intervals and 5% drop tests under t3 errors (n = 30, p = 2)."""
    rng = np.random.default_rng(20261118)
    covered = rejected = 0
    reps = 300
    for _ in range(reps):
        X = rng.normal(size=(30, 2))
        y = X[:, 0] + rng.standard_t(3, size=30)
        fit = RankRegressor().fit(X, y)
        covered += abs(fit.coef_[0] - 1) <= stats.t.ppf(0.975, 27) * fit.std_errors_[1]
        rejected += drop_test(fit, RankRegressor().fit(X[:, :1], y)).pvalue < 0.05
    # Exact binomial 99.9% bounds around the nominal levels.
    assert stats.binom.ppf(0.0005, reps, 0.95) <= covered <= stats.binom.ppf(0.9995, reps, 0.95)
    assert stats.binom.ppf(0.0005, reps, 0.05) <= rejected <= stats.binom.ppf(0.9995, reps, 0.05)


@pytest.mark.parametrize("name", R["regression"])
@pytest.mark.parametrize("implicit", [False, True])
def test_rfit_reference(name, implicit, monkeypatch):
    if implicit:
        monkeypatch.setattr(rank, "_TAU_IMPLICIT_MIN_SIZE", 0)
    r = R["regression"][name]
    X, y = np.array(r["X"]), np.array(r["y"])
    fit = RankRegressor().fit(X, y)
    # The exact minimum is never above Rfit's quasi-Newton value.
    assert fit.dispersion_ <= r["disp"] + 1e-9
    assert fit.dispersion_ == pytest.approx(r["disp"], rel=1e-5)
    np.testing.assert_allclose(np.r_[fit.intercept_, fit.coef_], r["coef"], atol=3e-3)
    residuals = np.array(r["residuals"])
    # On Rfit's own residuals, tau agrees within its root-finding tolerance.
    assert _tau(residuals, X.shape[1]) == pytest.approx(r["tauhat"], rel=1e-4)
    assert _tau_s(residuals, X.shape[1] + 1) == pytest.approx(r["taushat"], rel=1e-12)
    np.testing.assert_allclose(fit.std_errors_, r["se"], rtol=1e-2)
    assert drop_test(fit).statistic == pytest.approx(r["drop_F"], rel=1e-2)
    if "drop1_F" in r:
        reduced = RankRegressor().fit(X[:, :1], y)
        assert drop_test(fit, reduced).statistic == pytest.approx(r["drop1_F"], rel=1e-2)


def test_rank_anova_is_the_drop_test_of_the_one_way_model():
    rng = np.random.default_rng(9)
    groups = np.repeat(["x", "y", "z", "w"], 12)
    y = rng.standard_t(3, size=48) + np.repeat([0.0, 0.5, 0.0, 1.0], 12)
    levels = np.unique(groups)
    indicators = (groups[:, None] == levels[1:]).astype(float)
    assert rank_anova(y, groups) == drop_test(RankRegressor().fit(indicators, y))
    # The minimum is not unique with group indicators. The least squares choice
    # among the minimizers keeps the test free of the reference group and labels.
    renamed = np.array([dict(zip(levels, "dcba", strict=True))[g] for g in groups])
    for seed in range(10):
        y = np.random.default_rng(seed).standard_t(3, size=48)
        expected = rank_anova(y, groups)
        assert rank_anova(y, renamed).statistic == pytest.approx(expected.statistic, rel=1e-9)
    with pytest.raises(ValueError, match="two groups"):
        rank_anova(y, np.zeros(48))
    with pytest.raises(ValueError, match="shape"):
        rank_anova(y, groups[:-1])


def test_rank_anova_rfit_reference():
    r = R["oneway"]
    result = rank_anova(r["y"], r["g"])
    assert result.df == tuple(r["df"])
    # Rfit's tau comes from a numerical root with an absolute tolerance.
    assert result.statistic == pytest.approx(r["F"], rel=5e-3)
    assert result.pvalue == pytest.approx(r["p"], rel=2e-2)
