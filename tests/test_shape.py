"""Log-concave MLE and antitonic score matching: optimality, exact cases, equivariance, R values.

Reference values: tests/data/r_shape.json (R's logcondens and asm; its "meta" entry records the
versions). logcondens stops its Newton steps earlier: our log-likelihoods are higher, with the
same knots. asm's coefficients agree to about 1e-6.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels import shape
from robustmodels.shape import LogConcaveDensity, ScoreMatchingRegressor

R = json.loads((Path(__file__).parent / "data" / "r_shape.json").read_text())


def samples():
    rng = np.random.default_rng(0)
    return {
        "gamma": rng.gamma(3, size=150),
        "normal with ties": np.round(rng.normal(size=200), 1),
        "uniform": rng.uniform(size=80),
        "exponential": rng.exponential(size=120),
        "laplace": rng.laplace(size=100),
    }


def loglik(fit):
    x, phi = fit.support_, fit.log_density_
    return fit.weights_ @ phi - np.diff(x) @ shape._j00(phi[:-1], phi[1:])


@pytest.mark.parametrize("name", list(samples()))
def test_log_concave_mle_satisfies_its_characterization(name):
    # Dümbgen & Rufibach (2009): phi is concave, no concave kink at an observation increases the
    # likelihood, the density integrates to one, and its mean is the sample mean.
    x = samples()[name]
    fit = LogConcaveDensity().fit(x)
    support, phi = fit.support_, fit.log_density_
    knots = np.isin(support, fit.knots_)
    assert np.all(shape._kinks(support, phi) <= 1e-9)
    assert np.all(np.abs(shape._kinks(support, phi)[~knots]) <= 1e-9)
    assert shape._directional_derivatives(support, fit.weights_, phi, knots).max() <= 1e-9
    assert fit.cdf(support[-1])[0] == pytest.approx(1, abs=1e-12)
    grid = np.linspace(support[0], support[-1], 20001)
    density = np.exp(fit.score_samples(grid))
    mean = integrate.trapezoid(grid * density, grid)
    assert mean == pytest.approx(np.mean(x), abs=1e-6)
    variance = integrate.trapezoid((grid - mean) ** 2 * density, grid)
    assert variance <= np.var(x) + 1e-6
    cdf = integrate.cumulative_trapezoid(density, grid, initial=0)
    np.testing.assert_allclose(fit.cdf(grid), cdf, atol=1e-7)


def test_log_concave_exact_cases():
    fit = LogConcaveDensity().fit([1.0, 3.0])  # two points: uniform density
    np.testing.assert_allclose(np.exp(fit.score_samples([1, 2, 3])), 0.5)
    assert fit.score_samples([0.5, 3.5]).tolist() == [-np.inf, -np.inf]
    np.testing.assert_allclose(fit.cdf([0, 2, 4]), [0, 0.5, 1])
    # A normal sample: the estimate stays close to the normal density.
    x = np.random.default_rng(1).normal(size=2000)
    fit = LogConcaveDensity().fit(x)
    grid = np.linspace(-1.5, 1.5, 7)
    np.testing.assert_allclose(fit.score_samples(grid), -(grid**2) / 2 - 0.919, atol=0.12)


def test_integrals_of_linear_exponentials_match_quadrature():
    for a, b in [(0.3, 0.3), (0.3, 0.3005), (0.3, 0.315), (-1.0, 2.0), (2.0, -4.0)]:
        for function, weight in [
            (shape._j00, lambda t: 1),
            (shape._j10, lambda t: 1 - t),
            (shape._j11, lambda t: t * (1 - t)),
            (shape._j20, lambda t: (1 - t) ** 2),
        ]:
            integrand = lambda t, w=weight, a=a, b=b: w(t) * np.exp((1 - t) * a + t * b)  # noqa: E731
            expected = integrate.quad(integrand, 0, 1)[0]
            assert function(np.array([a]), np.array([b]))[0] == pytest.approx(expected, rel=1e-12)


def test_log_concave_matches_logcondens():
    for key, data in [("lc_gamma", "x_gamma"), ("lc_tied", "x_tied")]:
        fit, reference = LogConcaveDensity().fit(R[data]), R[key]
        np.testing.assert_array_equal(fit.knots_, reference["knots"])
        np.testing.assert_allclose(fit.log_density_, reference["phi"], atol=2e-4)
        assert loglik(fit) >= reference["loglik"]


def test_log_concave_input_checks():
    with pytest.raises(ValueError, match="two distinct"):
        LogConcaveDensity().fit([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        LogConcaveDensity().fit([1.0, np.nan])


@parametrize_with_checks([ScoreMatchingRegressor(), ScoreMatchingRegressor(symmetric=False)])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def regression(errors, n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    return X, 1 + X @ [2.0, -1.0] + errors(rng, n)


def test_score_is_decreasing_and_solves_the_estimating_equation():
    X, y = regression(lambda rng, n: rng.standard_t(2, n))
    fit = ScoreMatchingRegressor().fit(X, y)
    grid = np.linspace(-30, 30, 2001)
    psi = fit.psi(grid)
    assert np.all(np.diff(psi) <= 1e-12)
    np.testing.assert_allclose(fit.psi(-grid), -psi, atol=1e-12)
    residuals = y - fit.predict(X)
    design = np.column_stack([np.ones(len(y)), X])
    np.testing.assert_allclose(design.T @ fit.psi(residuals), 0, atol=1e-9)
    assert fit.information_ == pytest.approx(np.mean(fit.psi(residuals) ** 2))
    assert "z value" in fit.summary(["a", "b"])


@pytest.mark.parametrize("symmetric", [True, False])
def test_regression_and_scale_equivariance(symmetric):
    X, y = regression(lambda rng, n: rng.gamma(2, size=n))
    fit = ScoreMatchingRegressor(symmetric=symmetric).fit(X, y)
    beta = np.r_[fit.intercept_, fit.coef_]
    moved = ScoreMatchingRegressor(symmetric=symmetric).fit(X, 3 * y + X @ [1.0, 2.0] - 4)
    expected = 3 * beta + [-4, 1, 2]
    np.testing.assert_allclose(np.r_[moved.intercept_, moved.coef_], expected, atol=1e-6)
    np.testing.assert_allclose(moved.std_errors_, 3 * fit.std_errors_, rtol=1e-6)


def test_normal_errors_give_nearly_least_squares():
    # The normal score is already decreasing, so the fit is asymptotically least squares.
    X, y = regression(lambda rng, n: rng.normal(size=n), n=2000, seed=3)
    fit = ScoreMatchingRegressor().fit(X, y)
    design = np.column_stack([np.ones(len(y)), X])
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.all(np.abs(np.r_[fit.intercept_, fit.coef_] - ols) < 0.2 * fit.std_errors_)
    assert fit.information_ == pytest.approx(1 / np.var(y - design @ ols), rel=0.1)


def test_score_matching_matches_asm():
    X = np.array(R["X"])
    cases = [("asm_symmetric", True, "y_cauchy"), ("asm_skewed", False, "y_skewed")]
    for key, symmetric, y in cases:
        fit, reference = ScoreMatchingRegressor(symmetric=symmetric).fit(X, R[y]), R[key]
        np.testing.assert_allclose(np.r_[fit.intercept_, fit.coef_], reference["coef"], rtol=1e-5)
        np.testing.assert_allclose(fit.std_errors_, reference["se"], rtol=1e-4)
        assert fit.information_ == pytest.approx(reference["info"], rel=1e-3)


def test_newton_steps_fall_back_to_gradient_steps():
    # Huber loss (psi = -clip(r, -1, 1)) from a start where every residual is in the flat part.
    def huber(r):
        slope = -(np.abs(r) <= 1).astype(float)
        return -np.clip(r, -1, 1), slope, np.where(np.abs(r) <= 1, r**2 / 2, np.abs(r) - 0.5)

    y = 10 + np.random.default_rng(0).normal(size=50)
    beta = shape._minimize_loss(np.ones((50, 1)), y, np.zeros(1), huber)
    assert np.sum(np.clip(y - beta[0], -1, 1)) == pytest.approx(0, abs=1e-10)


def test_rule_of_thumb_bandwidth_matches_r():
    for x, expected in [
        ([0, 0, 0, 0, 1], 0.29171818740469724),
        ([0, 0, 0], 0.72246740558420763),
        ([3, 3, 3], 2.16740221675262301),
        ([1, 2, 4, 8, 16], 2.92075386855190722),
    ]:
        assert shape._nrd0(np.array(x, dtype=float)) == pytest.approx(expected, rel=1e-14)


def test_score_matching_input_checks():
    X, y = regression(lambda rng, n: rng.normal(size=n), n=20)
    with pytest.raises(ValueError, match="more observations"):
        ScoreMatchingRegressor().fit(X[:3], y[:3])
    with pytest.raises(ValueError, match="singular"):
        ScoreMatchingRegressor().fit(np.c_[X[:, 0], 2 * X[:, 0]], y)
