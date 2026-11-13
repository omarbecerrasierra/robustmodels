"""Estimating equations, numerical derivatives, contamination and R witnesses."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels.glm import (
    BYLogisticRegression,
    MallowsLogisticRegression,
    MallowsPoissonRegressor,
    MTPoissonRegressor,
    _by_loss,
    _mallows,
    _mallows_moments,
    _mt_mean,
    _MTProblem,
    _x_weights,
)

R = json.loads((Path(__file__).parent / "data" / "r_glm.json").read_text())
ESTIMATORS = [
    MallowsLogisticRegression,
    MallowsPoissonRegressor,
    BYLogisticRegression,
    MTPoissonRegressor,
]


@parametrize_with_checks(
    [
        MallowsLogisticRegression(),
        MallowsPoissonRegressor(),
        BYLogisticRegression(weighted=False),
        MTPoissonRegressor(n_trials=10, random_state=0),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("name", R["mallows"])
def test_mallows_r_coefficients_and_weights(name):
    r = R["mallows"][name]
    X, y, w = np.array(r["X"]), np.array(r["y"]), np.array(r["w"])
    design = np.c_[np.ones(len(y)), X]
    family = name.split("_")[0]
    coef, cov, weights, _, converged = _mallows(design, y, family, 1.345, w, 500, 1e-12)
    assert converged
    np.testing.assert_allclose(coef, r["coef"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(weights, r["wr"], atol=1e-9, rtol=0)
    # R uses a different correction in the meat; do not assert identical SEs.
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), r["se"], rtol=0.01)


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_mallows_moments_against_direct_summation(family):
    means = [0.001, 0.2, 0.8] if family == "binomial" else [0.001, 0.2, 3, 20, 100]
    for mu in means:
        values = np.arange(2) if family == "binomial" else np.arange(250)
        mass = (
            stats.binom.pmf(values, 1, mu)
            if family == "binomial"
            else stats.poisson.pmf(values, mu)
        )
        variance = mu * (1 - mu) if family == "binomial" else mu
        residuals = (values - mu) / np.sqrt(variance)
        psi = np.clip(residuals, -1.345, 1.345)
        expected = [mass @ psi, mass @ (psi * (values - mu)) / variance, mass @ psi**2]
        actual = _mallows_moments(np.array([mu]), 1.345, family)
        np.testing.assert_allclose(np.array(actual).ravel(), expected, atol=1e-12)


@pytest.mark.parametrize("name", ["clean", "leverage", "subset"])
def test_by_against_r(name):
    r = R["by"][name]
    X, y = np.array(r["X"]), np.array(r["y"])
    fit = BYLogisticRegression(weighted=False).fit(X, y)
    np.testing.assert_allclose(np.r_[fit.intercept_, fit.coef_], r["coef"], atol=1e-3, rtol=0)
    np.testing.assert_allclose(fit.std_errors_, r["se"], atol=3e-4, rtol=0)
    assert fit.objective_ <= r["objective"] + 1e-10
    assert fit.converged_
    design = np.c_[np.ones(len(y)), X]
    assert np.linalg.norm(design.T @ _by_loss(fit.decision_function(X), y, 0.5)[1]) < 1e-6


def test_weighted_by_removes_bad_leverage():
    r = R["by"]["leverage"]
    X, y = np.array(r["X"]), np.array(r["y"])
    fit = BYLogisticRegression().fit(X, y)
    assert np.all(fit.weights_[:5] == 0)
    np.testing.assert_allclose(
        np.r_[fit.intercept_, fit.coef_], R["by"]["subset"]["coef"], atol=1e-3
    )


def test_by_derivatives():
    s = np.array([-80.0, -5, -0.5, 0, 0.5, 5, 80])
    y = np.array([0, 1, 0, 1, 0, 1, 0])
    step = 1e-5
    _, first, second = _by_loss(s, y, 0.5)
    plus, minus = _by_loss(s + step, y, 0.5), _by_loss(s - step, y, 0.5)
    np.testing.assert_allclose(first, (plus[0] - minus[0]) / (2 * step), atol=1e-9)
    np.testing.assert_allclose(second, (plus[1] - minus[1]) / (2 * step), atol=1e-9)


@pytest.mark.parametrize("name", R["mt"])
def test_mt_against_r_and_objective_gradient(name):
    r = R["mt"][name]
    X, y = np.array(r["X"]), np.array(r["y"])
    fit = MTPoissonRegressor(n_trials=50, random_state=0).fit(X, y)
    coef = np.r_[fit.intercept_, fit.coef_]
    np.testing.assert_allclose(coef, r["coef"], atol=1e-3, rtol=0)
    np.testing.assert_allclose(fit.weights_, r["wr"], atol=2e-3, rtol=0)
    np.testing.assert_allclose(fit.std_errors_, r["se"], atol=4e-4, rtol=0)
    problem = _MTProblem(np.c_[np.ones(len(y)), X], y, np.ones(len(y)), 2.1)
    np.testing.assert_allclose(
        problem.gradient(coef), optimize.approx_fprime(coef, problem.objective, 1e-6), atol=2e-4
    )
    assert fit.converged_
    if name == "outliers":
        assert np.all(fit.weights_[:8] == 0)


def test_mt_mean_is_continuous_and_differentiable():
    m = _mt_mean(2.1)
    mu = np.array([0.15, 0.55, 4.4, 20.3, 77.7, 100.1, 200])
    h = 1e-5
    np.testing.assert_allclose(m(mu, 1), (m(mu + h) - m(mu - h)) / (2 * h), atol=2e-6)
    np.testing.assert_allclose(m(mu, 2), (m(mu + h, 1) - m(mu - h, 1)) / (2 * h), atol=2e-4)
    assert abs(m(np.array([100 - h]))[0] - m(np.array([100 + h]))[0]) < 2e-6


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_fitted_interface_and_input_validation(cls):
    kwargs = {"n_trials": 10, "random_state": 0} if cls is MTPoissonRegressor else {}
    fit = cls(**kwargs)
    X = np.random.default_rng(4).normal(size=(80, 2))
    y = np.random.default_rng(5).poisson(np.exp(1 + X[:, 0]))
    if "Logistic" in cls.__name__:
        y = np.where(y > 2, "yes", "no")
    with pytest.raises(NotFittedError):
        fit.predict(X)
    fit.fit(X, y)
    assert fit.predict(X).shape == (80,)
    assert fit.cov_params_.shape == (3, 3)
    np.testing.assert_allclose(fit.cov_params_, fit.cov_params_.T, atol=1e-10)
    assert np.linalg.eigvalsh(fit.cov_params_).min() >= -1e-10
    assert np.isfinite(fit.std_errors_).all()
    assert "Estimate" in fit.summary(["a", "b"])
    with pytest.raises(ValueError, match="feature names"):
        fit.summary(["a"])
    with pytest.raises(ValueError, match="singular"):
        fit.fit(np.ones((80, 2)), y)
    with pytest.raises(ValueError, match="more observations"):
        fit.fit(X[:2], [0, 1])
    with pytest.raises(ValueError, match="c must"):
        cls(c=0, **kwargs).fit(X, y)
    with pytest.raises(ValueError, match="max_iter"):
        cls(max_iter=0, **kwargs).fit(X, y)


@pytest.mark.parametrize("cls", [MallowsLogisticRegression, BYLogisticRegression])
def test_binary_response_and_no_intercept(cls):
    r = R["mallows"]["binomial"]
    X, y = np.array(r["X"]), np.array(r["y"])
    fit = cls(fit_intercept=False).fit(X, y)
    assert fit.intercept_ == 0
    assert fit.std_errors_.shape == (2,)
    np.testing.assert_allclose(fit.predict_proba(X).sum(axis=1), 1)
    with pytest.raises(ValueError, match="2 classes"):
        fit.fit(X, np.zeros(len(y)))
    with pytest.raises(ValueError):
        fit.fit(X, np.arange(len(y)) / 5)


@pytest.mark.parametrize("cls", [MallowsPoissonRegressor, MTPoissonRegressor])
def test_poisson_response_and_no_intercept(cls):
    kwargs = {"n_trials": 10, "random_state": 0} if cls is MTPoissonRegressor else {}
    X = np.random.default_rng(3).normal(size=(60, 1))
    y = np.random.default_rng(4).poisson(np.exp(0.5 * X[:, 0]))
    fit = cls(fit_intercept=False, **kwargs).fit(X, y)
    assert fit.intercept_ == 0
    assert (fit.predict(X) > 0).all()
    with pytest.raises(ValueError, match="all-zero"):
        cls(**kwargs).fit(X, np.zeros(len(y)))
    with pytest.raises(ValueError, match="nonnegative"):
        fit.fit(X, -np.ones(len(y)))


def test_x_weights_and_nonconvergence():
    X = np.random.default_rng(0).normal(size=(50, 2))
    X[0] = 20
    design = np.c_[np.ones(50), X]
    np.testing.assert_allclose(
        _x_weights("hat", X, design), np.sqrt(1 - np.diag(design @ np.linalg.pinv(design)))
    )
    assert _x_weights("mcd", X, design)[0] < 0.05
    with pytest.raises(ValueError, match="x_weights"):
        _x_weights("invalid", X, design)
    y = np.random.default_rng(1).poisson(np.exp(1 + np.clip(X[:, 0], -2, 2)))
    for cls, kwargs in [
        (MallowsPoissonRegressor, {}),
        (MTPoissonRegressor, {"n_trials": 10}),
        (BYLogisticRegression, {"weighted": False}),
    ]:
        response = (y > 3).astype(int) if cls is BYLogisticRegression else y
        with pytest.warns(RuntimeWarning, match="did not converge"):
            cls(max_iter=1, **kwargs).fit(X, response)
    for tol in [0, -1, np.nan]:
        with pytest.raises(ValueError, match="tol"):
            MallowsPoissonRegressor(tol=tol).fit(X, y)


def test_mt_mean_small_tuning_does_not_select_wrong_local_mode():
    # At mu=0.1 and c=0.1 the support intervals around sqrt(count) are
    # disjoint and the mass at zero is largest. Its exact minimum is m=0.
    assert _mt_mean(0.1)(np.array([0.1]))[0] == pytest.approx(0, abs=1e-8)


def test_weighted_by_rejects_subset_without_both_classes():
    X = np.array(R["by"]["leverage"]["X"])
    y = np.zeros(len(X), dtype=int)
    y[:5] = 1  # all observations of the positive class are trimmed leverage points
    with pytest.raises(ValueError, match="weighted subset"):
        BYLogisticRegression().fit(X, y)


@pytest.mark.parametrize(
    "response, message", [([[1, 2]], "1-D response"), (["nan", "1"], "finite")]
)
def test_count_response_conversion_validation(response, message):
    # Validate conversion before any moments or optimization can see invalid counts.
    with pytest.raises(ValueError, match=message):
        MallowsPoissonRegressor()._counts(response)
