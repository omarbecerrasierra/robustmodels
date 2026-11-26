import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels.regression import (
    LTSRegressor,
    MMRegressor,
    SRegressor,
    _lts_small_sample,
    _positive_semidefinite,
    _s_scale,
    mm_test,
)
from robustmodels.scale import _rho, _tuning_for_breakdown, _tuning_for_efficiency


def expected_failures(estimator):
    # The check sets alpha=0.01, assuming a regularization strength; here alpha
    # is the LTS coverage in [0.5, 1], as in MCD and robustbase.
    if isinstance(estimator, LTSRegressor):
        return {"check_regressors_train": "alpha is the coverage, not a penalty"}
    return {}


@parametrize_with_checks(
    [
        LTSRegressor(random_state=0),
        LTSRegressor(fit_intercept=False, random_state=0),
        SRegressor(n_trials=50, random_state=0),
        MMRegressor(n_trials=50, random_state=0),
    ],
    expected_failed_checks=expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def contaminated(seed=0, n=60):
    """y = 1 + x1 + 2 x2 + noise; 9 bad leverage points and 9 vertical outliers."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 1 + X @ [1.0, 2.0] + rng.normal(scale=0.5, size=n)
    X[:9] += 6
    y[9:18] += 20
    return X, y


def least_squares(X, y):
    design = np.column_stack([np.ones(len(y)), X])
    coef, rss = np.linalg.lstsq(design, y, rcond=None)[:2]
    return coef, rss[0]


def test_fast_lts_finds_the_global_optimum():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 1))
    y = 2 * X[:, 0] + rng.normal(size=12)
    y[:4] += 8
    f = LTSRegressor(reweighting=False, random_state=0).fit(X, y)
    exhaustive = min(
        least_squares(X[list(s)], y[list(s)])[1] for s in itertools.combinations(range(12), f.h_)
    )
    objective = np.sort((y - f.predict(X)) ** 2)[: f.h_].sum()
    assert objective == pytest.approx(exhaustive, rel=1e-10)


def test_regression_scale_and_affine_equivariance():
    X, y = contaminated()
    A = np.array([[2.0, 1.0], [0.5, -1.0]])
    b = np.array([3.0, -2.0])
    base = LTSRegressor(random_state=0).fit(X, y)
    moved = LTSRegressor(random_state=0).fit(X @ A, -3 * (y + X @ b) + 5)
    np.testing.assert_allclose(moved.coef_, np.linalg.solve(A, -3 * (base.coef_ + b)))
    assert moved.intercept_ == pytest.approx(-3 * base.intercept_ + 5)
    assert moved.scale_ == pytest.approx(3 * base.scale_)
    np.testing.assert_array_equal(moved.support_, base.support_)


def test_resists_outliers_where_least_squares_fails():
    X, y = contaminated()
    lts = LTSRegressor(random_state=0).fit(X, y)
    np.testing.assert_allclose(np.r_[lts.intercept_, lts.coef_], [1, 1, 2], atol=0.3)
    assert not lts.support_[:18].any()
    assert np.abs(least_squares(X, y)[0] - [1, 1, 2]).max() > 2


def test_scale_is_unbiased_at_the_normal_model():
    rng = np.random.default_rng(3)
    scales = []
    for _ in range(200):
        X = rng.normal(size=(50, 2))
        y = X.sum(axis=1) + rng.normal(size=50)
        f = LTSRegressor(n_trials=30, random_state=0).fit(X, y)
        scales.append([f.raw_scale_, f.scale_])
    assert np.mean(scales, axis=0) == pytest.approx([1, 1], abs=0.04)


def test_full_coverage_is_least_squares():
    X, y = contaminated()
    f = LTSRegressor(alpha=1, reweighting=False).fit(X, y)
    coef, rss = least_squares(X, y)
    np.testing.assert_allclose(np.r_[f.intercept_, f.coef_], coef)
    assert f.scale_ == pytest.approx(np.sqrt(rss / (len(y) - 3)))
    assert f.h_ == len(y) and f.support_.all()


def test_reweighting_that_keeps_every_point_is_least_squares():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 1))
    y = 2 * X[:, 0] + rng.normal(scale=0.3, size=30)
    f = LTSRegressor(random_state=0).fit(X, y)
    coef, rss = least_squares(X, y)
    assert f.support_.all()
    np.testing.assert_allclose(np.r_[f.intercept_, f.coef_], coef)
    assert f.scale_ == pytest.approx(np.sqrt(rss / (len(y) - 1)))


def test_exact_fit_warns_and_keeps_the_raw_fit():
    X = np.arange(20.0)[:, None]
    y = 3 * X[:, 0] + 1
    y[:5] = [40, -7, 3, 100, 8]
    with pytest.warns(RuntimeWarning, match="exact fit"):
        f = LTSRegressor(random_state=0).fit(X, y)
    assert f.scale_ == 0
    assert (f.intercept_, f.coef_[0]) == pytest.approx((1, 3))
    np.testing.assert_array_equal(f.support_, np.arange(20) >= 5)


def test_a_gross_outlier_is_not_mistaken_for_an_exact_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 1))
    y = 1 + 2 * X[:, 0] + rng.normal(scale=0.5, size=50)
    y[0] = 1e12
    f = LTSRegressor(random_state=0).fit(X, y)
    assert 0.3 < f.scale_ < 0.8
    assert not f.support_[0]


def test_small_sample_correction_multiplies_the_raw_scale():
    X, y = contaminated()
    on = LTSRegressor(random_state=0).fit(X, y)
    off = LTSRegressor(small_sample_correction=False, random_state=0).fit(X, y)
    factor = _lts_small_sample(2, 60, 0.5, True, False)
    assert on.raw_scale_ == pytest.approx(off.raw_scale_ * factor)


def test_small_sample_factor_limits():
    assert _lts_small_sample(2, 50, 1.0, True, False) == 1.0  # alpha = 1
    assert _lts_small_sample(1, 2, 0.5, True, True) == 1.0  # fitted curve <= 0


@pytest.mark.parametrize(
    "estimator, n, message",
    [
        (LTSRegressor(alpha=0.3), 20, "alpha"),
        (LTSRegressor(n_trials=0), 20, "n_trials"),
        (LTSRegressor(), 3, "more observations than coefficients"),
    ],
)
def test_invalid_input_raises_a_clear_error(estimator, n, message):
    X = np.random.default_rng(5).normal(size=(n, 2))
    with pytest.raises(ValueError, match=message):
        estimator.fit(X, X.sum(axis=1))


def test_collinear_columns_raise_a_clear_error():
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="singular"):
        LTSRegressor().fit(np.column_stack([x, 2 * x]), x)


# S and MM regression

C0 = _tuning_for_breakdown(0.5)


def mean_loss(X, y, fit, c):
    return np.mean(_rho((y - fit.predict(X)) / fit.scale_, c))


@pytest.mark.parametrize("estimator", [SRegressor, MMRegressor])
def test_s_and_mm_resist_outliers(estimator):
    X, y = contaminated()
    fit = estimator(random_state=0).fit(X, y)
    np.testing.assert_allclose(np.r_[fit.intercept_, fit.coef_], [1, 1, 2], atol=0.3)


def test_s_scale_solves_its_equation_and_is_minimal():
    X, y = contaminated()
    s = SRegressor(random_state=0).fit(X, y)
    n, p = len(y), 3
    assert np.sum(_rho((y - s.predict(X)) / s.scale_, C0)) / (n - p) == pytest.approx(0.5)
    lts = LTSRegressor(random_state=0).fit(X, y)
    ols = least_squares(X, y)[0]
    truth = y - 1 - X @ [1.0, 2.0]
    for residuals in (y - lts.predict(X), truth, y - ols[0] - X @ ols[1:]):
        assert s.scale_ <= _s_scale(residuals, C0, 0.5, p, 0.0)


def test_mm_starts_from_s_and_lowers_its_loss():
    X, y = contaminated()
    s = SRegressor(random_state=0).fit(X, y)
    mm = MMRegressor(random_state=0).fit(X, y)
    np.testing.assert_allclose(np.r_[mm.s_intercept_, mm.s_coef_], np.r_[s.intercept_, s.coef_])
    assert mm.scale_ == s.scale_
    c = _tuning_for_efficiency(0.95)
    assert mean_loss(X, y, mm, c) <= mean_loss(X, y, s, c)


@pytest.mark.parametrize("estimator", [SRegressor, MMRegressor])
def test_s_and_mm_equivariance(estimator):
    X, y = contaminated()
    A = np.array([[2.0, 1.0], [0.5, -1.0]])
    b = np.array([3.0, -2.0])
    base = estimator(random_state=0).fit(X, y)
    moved = estimator(random_state=0).fit(X @ A, -3 * (y + X @ b) + 5)
    np.testing.assert_allclose(moved.coef_, np.linalg.solve(A, -3 * (base.coef_ + b)), rtol=1e-6)
    assert moved.intercept_ == pytest.approx(-3 * base.intercept_ + 5, rel=1e-6)
    assert moved.scale_ == pytest.approx(3 * base.scale_, rel=1e-6)


def test_gaussian_efficiency_and_scale_consistency():
    rng = np.random.default_rng(7)
    slopes = {"ols": [], "s": [], "mm": []}
    scales, errors = [], []
    for _ in range(200):
        X = rng.normal(size=(100, 2))
        y = X @ [1.0, 1.0] + rng.normal(size=100)
        slopes["ols"].append(least_squares(X, y)[0][1:])
        mm = MMRegressor(n_trials=10, random_state=0).fit(X, y)  # s_coef_: the S fit
        slopes["s"].append(mm.s_coef_)
        slopes["mm"].append(mm.coef_)
        scales.append(mm.scale_)
        errors.append(mm.std_errors_[1:])
    variance = {k: np.var(v, axis=0).mean() for k, v in slopes.items()}
    assert variance["ols"] / variance["mm"] == pytest.approx(0.95, abs=0.05)
    assert variance["ols"] / variance["s"] == pytest.approx(0.287, abs=0.08)
    assert np.mean(scales) == pytest.approx(1, abs=0.04)
    # standard errors match the spread of the estimates (sandwich: slightly low)
    ratio = np.mean(errors, axis=0) / np.std(slopes["mm"], axis=0)
    assert np.all((0.8 < ratio) & (ratio < 1.2))


def test_mm_confidence_intervals_hold_under_heteroskedasticity():
    rng = np.random.default_rng(11)
    q = stats.t.ppf(0.975, 100 - 3)
    covered = []
    for _ in range(100):
        X = rng.normal(size=(100, 2))
        y = X @ [1.0, 1.0] + rng.normal(size=100) * np.exp(X[:, 0]) / 1.6
        mm = MMRegressor(n_trials=10, random_state=0).fit(X, y)
        covered.append(np.abs(mm.coef_ - 1) <= q * mm.std_errors_[1:])
    assert np.mean(covered) > 0.9


def test_mm_inference_attributes_and_summary():
    X, y = contaminated()
    mm = MMRegressor(random_state=0).fit(X, y)
    estimates = np.r_[mm.intercept_, mm.coef_]
    np.testing.assert_allclose(mm.std_errors_, np.sqrt(np.diag(mm.cov_params_)))
    np.testing.assert_allclose(mm.t_values_, estimates / mm.std_errors_)
    np.testing.assert_allclose(mm.p_values_, 2 * stats.t.sf(np.abs(mm.t_values_), len(y) - 3))
    lines = mm.summary(["height", "weight"]).splitlines()
    assert lines[0].split() == ["Estimate", "Std.", "Error", "t", "value", "Pr(>|t|)"]
    assert [line.split()[0] for line in lines[1:4]] == ["(Intercept)", "height", "weight"]
    assert lines[-1].startswith("Residual scale:")
    assert "x1" in MMRegressor(fit_intercept=False, random_state=0).fit(X, y).summary()
    with pytest.raises(ValueError, match="feature names"):
        mm.summary(["only one"])


def test_covariance_fix_keeps_the_variances():
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    fixed = _positive_semidefinite(indefinite)
    assert np.linalg.eigvalsh(fixed).min() > -1e-12
    np.testing.assert_allclose(np.diag(fixed), [1, 1])
    identity = np.eye(2)
    assert _positive_semidefinite(identity) is identity


@pytest.mark.parametrize("estimator", [SRegressor, MMRegressor])
def test_s_and_mm_exact_fit_warns(estimator):
    X = np.arange(20.0)[:, None]
    y = 3 * X[:, 0] + 1
    y[:5] = [40, -7, 3, 100, 8]
    with pytest.warns(RuntimeWarning, match="exact fit"):
        fit = estimator(random_state=0).fit(X, y)
    assert fit.scale_ == 0
    assert (fit.intercept_, fit.coef_[0]) == pytest.approx((1, 3))
    if estimator is MMRegressor:
        assert np.isnan(fit.std_errors_).all()


def test_iteration_limits_name_the_stage():
    X, y = contaminated()
    with pytest.warns(RuntimeWarning, match="S did not converge"):
        SRegressor(max_iter=1, random_state=0).fit(X, y)
    with pytest.warns(RuntimeWarning) as record:
        MMRegressor(max_iter=1, random_state=0).fit(X, y)
    messages = {str(w.message) for w in record}
    assert {"S did not converge within max_iter", "MM did not converge within max_iter"} <= messages


@pytest.mark.parametrize(
    "estimator, message",
    [
        (SRegressor(breakdown=0.7), "breakdown"),
        (SRegressor(n_best=0), "n_best"),
        (SRegressor(k_steps=0), "k_steps"),
        (MMRegressor(efficiency=1.5), "efficiency"),
        (MMRegressor(efficiency=0.2), "efficiency is too low"),
        (MMRegressor(max_iter=0), "max_iter"),
    ],
)
def test_invalid_s_and_mm_parameters(estimator, message):
    X = np.random.default_rng(5).normal(size=(20, 2))
    with pytest.raises(ValueError, match=message):
        estimator.fit(X, X.sum(axis=1))


ANOVA = json.loads((Path(__file__).parent / "data" / "r_reference.json").read_text())["anova"]


@pytest.mark.parametrize("name", ANOVA)
def test_mm_test_robustbase_reference(name):
    """robustbase's anova.lmrob; its rounded bisquare constants move the fit by about 1e-6."""
    r = ANOVA[name]
    X, y = np.array(r["X"]), np.array(r["y"])
    mm = MMRegressor(random_state=0).fit(X, y)
    np.testing.assert_allclose(np.r_[mm.intercept_, mm.coef_], r["coef"], atol=1e-4)
    for case in r["tests"]:
        result = mm_test(mm, X, y, case["drop"], test=case["test"])
        assert result.statistic == pytest.approx(case["statistic"], rel=2e-3)
        assert result.pvalue == pytest.approx(case["pvalue"], rel=2e-2, abs=1e-12)
        assert result.df == len(case["drop"])


def test_mm_test_definitions_and_invariance():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(70, 3))
    y = 1 + X[:, 0] + rng.standard_t(3, size=70)
    mm = MMRegressor(random_state=0).fit(X, y)
    wald = mm_test(mm, X, y, [1, 2], test="wald")
    block = mm.cov_params_[np.ix_([2, 3], [2, 3])]
    assert wald.statistic == pytest.approx(mm.coef_[1:] @ np.linalg.solve(block, mm.coef_[1:]))
    assert wald.pvalue == pytest.approx(stats.chi2.sf(wald.statistic, 2))
    deviance = mm_test(mm, X, y, [1, 2])
    # Invariant to the scale of the response.
    scaled = MMRegressor(random_state=0).fit(X, 5 * y)
    assert mm_test(scaled, X, 5 * y, [1, 2]).statistic == pytest.approx(deviance.statistic)
    # A reduced model with no column at all (no intercept, every slope dropped).
    origin = MMRegressor(fit_intercept=False, random_state=0).fit(X, y)
    assert mm_test(origin, X, y, [0, 1, 2]).pvalue < 1e-4


def test_mm_test_size_under_the_null():
    """Monte Carlo: 5% tests with normal and t3 errors (n = 60, two slopes tested)."""
    rng = np.random.default_rng(20261118)
    reps, rejected = 150, {"wald": 0, "deviance": 0}
    for k in range(reps):
        X = rng.normal(size=(60, 3))
        errors = rng.normal(size=60) if k % 2 else rng.standard_t(3, size=60)
        y = X[:, 0] + errors
        mm = MMRegressor(n_trials=100, random_state=k).fit(X, y)
        for test in rejected:
            rejected[test] += mm_test(mm, X, y, [1, 2], test=test).pvalue < 0.05
    for count in rejected.values():
        assert stats.binom.ppf(0.0005, reps, 0.05) <= count <= stats.binom.ppf(0.9995, reps, 0.05)


def test_mm_test_input_checks():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + rng.normal(size=30)
    mm = MMRegressor(n_trials=50, random_state=0).fit(X, y)
    assert mm_test(mm, X, y, 0) == mm_test(mm, X, y, [0])
    with pytest.raises(ValueError, match="MMRegressor"):
        mm_test(SRegressor(n_trials=50, random_state=0).fit(X, y), X, y, [0])
    with pytest.raises(ValueError, match="test must be"):
        mm_test(mm, X, y, [0], test="lrt")
    with pytest.raises(ValueError, match="columns"):
        mm_test(mm, X, y, [2])
    with pytest.raises(ValueError, match="columns"):
        mm_test(mm, X, y, [0.9])
    with pytest.raises(ValueError, match="training data"):
        mm_test(mm, X, y[::-1], [0])
    with pytest.raises(ValueError, match="training data"):
        mm_test(mm, X + 0.5, y, [0], test="wald")
    exact = MMRegressor(n_trials=50, random_state=0).fit(X, 1 + X[:, 0])
    with pytest.raises(ValueError, match="exact fit"):
        mm_test(exact, X, 1 + X[:, 0], [1])
    with pytest.raises(ValueError, match="not available"):
        mm_test(exact, X, 1 + X[:, 0], [1], test="wald")


def _scaled_fits():
    from robustmodels.glm import (
        BYLogisticRegression,
        MallowsLogisticRegression,
        MallowsPoissonRegressor,
        MTPoissonRegressor,
    )
    from robustmodels.rank import RankRegressor
    from robustmodels.shape import ScoreMatchingRegressor

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(150, 2))
    eta = 0.3 + Z @ [0.8, -0.5]
    binary = (rng.uniform(size=150) < 1 / (1 + np.exp(-eta))).astype(int)
    counts, linear = rng.poisson(np.exp(eta)), 1 + Z @ [2.0, -1.0] + rng.standard_t(3, size=150)
    return Z, [
        (LTSRegressor(random_state=0), linear),
        (SRegressor(random_state=0), linear),
        (MMRegressor(random_state=0), linear),
        (RankRegressor(), linear),
        (ScoreMatchingRegressor(), linear),
        (MallowsLogisticRegression(), binary),
        (MallowsPoissonRegressor(), counts),
        (BYLogisticRegression(), binary),
        (MTPoissonRegressor(n_trials=50, random_state=0), counts),
    ]


def test_fits_are_equivariant_to_large_means_and_disparate_scales():
    Z, cases = _scaled_fits()
    scale = np.array([1e6, 1e-6])
    for model, y in cases:
        base = model.fit(Z, y)
        coef, errors = base.coef_.copy(), getattr(base, "std_errors_", None)
        shifted = model.fit((Z + 1e7) * scale, y)  # was rejected as singular or wrong
        np.testing.assert_allclose(shifted.coef_ * scale, coef, rtol=1e-6, err_msg=str(model))
        if errors is not None:
            np.testing.assert_allclose(
                shifted.std_errors_[-2:] * scale, errors[-2:], rtol=1e-6, err_msg=str(model)
            )
