import itertools

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels.regression import LTSRegressor, _lts_small_sample


def expected_failures(estimator):
    # The check sets alpha=0.01, assuming a regularization strength; here alpha
    # is the LTS coverage in [0.5, 1], as in MCD and robustbase.
    return {"check_regressors_train": "alpha is the coverage, not a penalty"}


@parametrize_with_checks(
    [LTSRegressor(random_state=0), LTSRegressor(fit_intercept=False, random_state=0)],
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
