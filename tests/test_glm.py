"""Estimating equations, numerical derivatives, contamination and R witnesses."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats
from scipy.special import expit
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


def test_mt_mean_rejects_tuning_that_cannot_identify_small_means():
    # For c <= 1, positive counts lie outside the score's support at m=0:
    # m=0 is a stationary minimum for a range of means and cannot be strictly
    # increasing, as required by the MT consistency assumptions.
    with pytest.raises(ValueError, match="c must be in"):
        _mt_mean(0.1)


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


def test_mt_transformation_by_direct_poisson_score_at_small_means():
    from robustmodels.regression import _bisquare

    means = np.array([0.001, 0.01, 0.05, 0.1, 0.15, 0.25, 0.55, 1.55])
    transformed = _mt_mean(2.1)(means)
    scores = []
    for mu, location in zip(means, transformed, strict=True):
        values = np.arange(40)
        psi = _bisquare(np.sqrt(values) - location, 2.1)[0]
        scores.append(stats.poisson.pmf(values, mu) @ psi)
    np.testing.assert_allclose(scores, 0, atol=3e-6, rtol=0)
    assert np.all(np.diff(transformed) > 0)


def test_mt_clipped_objective_has_zero_gradient_and_no_inference():
    # Choose observations within the nonconstant part of the bisquare loss;
    # otherwise a zero derivative would occur from rejection alone.
    mu = np.exp(30)
    y = np.full(3, np.round((np.sqrt(mu) + 0.5) ** 2))
    problem = _MTProblem(np.ones((3, 1)), y, np.ones(3), 2.1)
    coef = np.array([30.1])
    h = 1e-5
    numeric = (problem.objective(coef + h) - problem.objective(coef - h)) / (2 * h)
    np.testing.assert_allclose(problem.gradient(coef), [numeric], atol=1e-10)
    assert problem.is_clipped(coef)
    assert np.isnan(problem.covariance(coef)).all()


def test_mt_numerical_plateau_cannot_report_convergence(monkeypatch):
    from types import SimpleNamespace

    from robustmodels import glm

    monkeypatch.setattr(
        glm.optimize,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.array([31.0, 0.0]), fun=1.0, nit=1
        ),
    )
    X = np.arange(10.0)[:, None]
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = MTPoissonRegressor(n_trials=2, random_state=0).fit(X, np.ones(10))
    assert not fit.converged_
    assert np.isnan(fit.std_errors_).all()


def test_by_score_has_zero_conditional_expectation():
    predictors = np.linspace(-6, 6, 31)
    probability = 1 / (1 + np.exp(-predictors))
    positive = _by_loss(predictors, np.ones(len(predictors)), 0.5)[1]
    negative = _by_loss(predictors, np.zeros(len(predictors)), 0.5)[1]
    np.testing.assert_allclose(probability * positive + (1 - probability) * negative, 0, atol=1e-16)


def test_mallows_covariance_from_conditional_distribution_derivatives():
    rng = np.random.default_rng(15)
    X = np.linspace(-1, 1, 60)[:, None]
    design = np.c_[np.ones(len(X)), X]
    y = rng.poisson(np.exp(design @ [0.3, 0.5]))
    fit = MallowsPoissonRegressor(tol=1e-11).fit(X, y)
    eta = fit.intercept_ + X[:, 0] * fit.coef_[0]
    values = np.arange(100)
    c, h = fit.c, 1e-5
    sensitivity, variance = [], []
    for linear in eta:
        mu = np.exp(linear)
        true_mass = stats.poisson.pmf(values, mu)

        def centered_score(candidate):
            mean = np.exp(candidate)
            psi = np.clip((values - mean) / np.sqrt(mean), -c, c)
            correction = stats.poisson.pmf(values, mean) @ psi
            return (psi - correction) * np.sqrt(mean)

        sensitivity.append(
            -true_mass @ (centered_score(linear + h) - centered_score(linear - h)) / (2 * h)
        )
        variance.append(true_mass @ centered_score(linear) ** 2)
    bread = np.linalg.inv(design.T @ (design * np.array(sensitivity)[:, None]))
    meat = design.T @ (design * np.array(variance)[:, None])
    np.testing.assert_allclose(fit.cov_params_, bread @ meat @ bread.T, rtol=1e-7)


def test_mallows_poisson_with_exposure_solves_the_corrected_score():
    rng = np.random.default_rng(96)
    X = rng.normal(size=(160, 1))
    exposure = rng.uniform(0.2, 3, size=len(X))
    y = rng.poisson(exposure * np.exp(0.5 + 0.3 * X[:, 0]))
    model = MallowsPoissonRegressor(tol=1e-11).fit(X, y, offset=np.log(exposure))
    mu = model.predict(X, offset=np.log(exposure))
    np.testing.assert_allclose(mu, exposure * model.predict(X))
    counts = np.arange(100)
    roots = np.sqrt(mu)
    score = np.clip((y - mu) / roots, -model.c, model.c)
    mass = stats.poisson.pmf(counts, mu[:, None])
    possible = np.clip((counts - mu[:, None]) / roots[:, None], -model.c, model.c)
    corrected = (score - (mass * possible).sum(axis=1)) * roots
    np.testing.assert_allclose(np.c_[np.ones(len(X)), X].T @ corrected, 0, atol=1e-8)
    ordinary = MallowsPoissonRegressor().fit(X, y)
    constant = MallowsPoissonRegressor().fit(X, y, offset=2)
    np.testing.assert_allclose(constant.predict(X, offset=2), ordinary.predict(X), rtol=1e-8)
    assert constant.intercept_ + 2 == pytest.approx(ordinary.intercept_)
    for bad in ([0, 1], np.inf):
        with pytest.raises(ValueError, match="offset"):
            model.predict(X, offset=bad)


def test_mallows_poisson_cap_does_not_report_inference():
    X = np.arange(8.0)[:, None]
    offset = np.tile([-100.0, 100.0], 4)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = MallowsPoissonRegressor(max_iter=2).fit(X, np.ones(len(X)), offset=offset)
    assert not fit.converged_
    assert np.isnan(fit.std_errors_).all()


def test_logistic_probability_of_the_rare_class_is_retained():
    X = np.arange(8.0)[:, None]
    model = MallowsLogisticRegression().fit(X, [0, 0, 1, 0, 1, 1, 0, 1])
    point = [[(50 - model.intercept_) / model.coef_[0]]]
    probability = model.predict_proba(point)[0, 0]
    assert probability > 0
    assert probability == pytest.approx(1 / (1 + np.exp(50)), abs=0)


def test_mallows_reports_inference_when_only_negligible_scores_are_capped():
    # A leverage point fitted with |eta| > 30: its score contribution is ~0 either way.
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(size=200), 40.0]
    y = (rng.uniform(size=x.size) < expit(0.5 + x)).astype(int)
    logistic = MallowsLogisticRegression().fit(x[:, None], y)
    z = np.r_[rng.normal(size=150), 40.0]
    counts = rng.poisson(np.exp(1 - z))
    poisson = MallowsPoissonRegressor().fit(z[:, None], counts)
    for model, values in ((logistic, x), (poisson, z)):
        assert np.max(np.abs(model._linear_predictor(values[:, None]))) > 30
        assert model.converged_
        assert np.isfinite(model.std_errors_).all()
