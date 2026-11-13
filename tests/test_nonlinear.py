"""Nonlinear M regression: R coefficients and independent covariance calculation."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from robustmodels.regression import NonlinearMRegressor

R = json.loads((Path(__file__).parent / "data" / "r_nl.json").read_text())


def decay(X, a, b):
    return a * np.exp(-b * X[:, 0])


def logistic(X, a, m, s):
    return a / (1 + np.exp((m - np.log(X[:, 0])) / s))


@pytest.mark.parametrize(
    "name, model, start", [("decay", decay, [5, 1]), ("dnase", logistic, [3, 0, 1])]
)
def test_nlrob_reference(name, model, start):
    r = R[name]
    X, y = np.array(r["x"])[:, None], np.array(r["y"])
    original = start.copy()
    fit = NonlinearMRegressor(model, start, tol=1e-10, max_iter=200).fit(X, y)
    np.testing.assert_allclose(fit.params_, r["coef"], atol=3e-5, rtol=0)
    np.testing.assert_allclose(fit.weights_, r["w"], atol=2e-5, rtol=0)
    assert fit.scale_ == pytest.approx(r["scale"], abs=4e-6)
    residuals = y - fit.predict(X)
    assert fit.scale_ == pytest.approx(np.median(np.abs(residuals)) / 0.6745)
    assert fit.converged_
    assert start == original
    assert clone(fit).start == original


def test_nonlinear_covariance_against_analytic_jacobian():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, size=(80, 1))
    y = decay(X, 10, 0.7) + rng.normal(scale=0.3, size=80)
    y[:8] += 15
    fit = NonlinearMRegressor(decay, [5, 1]).fit(X, y)
    a, b = fit.params_
    jacobian = np.c_[np.exp(-b * X[:, 0]), -a * X[:, 0] * np.exp(-b * X[:, 0])]
    t = (y - fit.predict(X)) / fit.scale_
    psi = np.clip(t, -fit.c, fit.c)
    factor = fit.scale_**2 * np.sum(psi**2) / (len(y) - 2) / np.mean(np.abs(t) <= fit.c) ** 2
    expected = factor * np.linalg.inv(jacobian.T @ jacobian)
    np.testing.assert_allclose(fit.cov_params_, expected, rtol=1e-6)
    assert fit.weights_[:8].max() < 0.1
    np.testing.assert_allclose(fit.params_, [10, 0.7], atol=0.35)


def test_exact_fit_assigns_zero_weight_to_nonzero_residuals():
    def constant(X, a):
        return np.full(len(X), a)

    X = np.arange(10.0)[:, None]
    y = np.r_[np.zeros(8), 10, 20]
    with pytest.warns(RuntimeWarning, match="exact fit"):
        fit = NonlinearMRegressor(constant, [0.0]).fit(X, y)
    np.testing.assert_array_equal(fit.weights_, [1] * 8 + [0, 0])
    np.testing.assert_array_equal(fit.cov_params_, [[0.0]])
    assert fit.scale_ == 0


@pytest.mark.parametrize("start", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_start(start):
    with pytest.raises(ValueError, match="start"):
        NonlinearMRegressor(decay, start).fit(np.ones((5, 1)), np.ones(5))


def test_validation_nonconvergence_and_prediction():
    X = np.arange(10.0)[:, None]
    y = np.exp(-X[:, 0]) + 0.01
    with pytest.raises(NotFittedError):
        NonlinearMRegressor(decay, [1, 1]).predict(X)
    with pytest.raises(ValueError, match="model must be callable"):
        NonlinearMRegressor(None, [1]).fit(X, y)
    for model in [
        lambda X, a: 1,
        lambda X, a: np.ones((len(X), 1)),
        lambda X, a: np.full(len(X), np.nan),
    ]:
        with pytest.raises(ValueError, match="model must return"):
            NonlinearMRegressor(model, [1]).fit(X, y)
    with pytest.raises(ValueError, match="more observations"):
        NonlinearMRegressor(decay, [1, 1]).fit(X[:2], y[:2])
    for key in ["tol", "c", "max_iter"]:
        with pytest.raises(ValueError, match=key):
            NonlinearMRegressor(decay, [1, 1], **{key: 0}).fit(X, y)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = NonlinearMRegressor(decay, [2, 2], max_iter=1).fit(X, y)
    assert not fit.converged_
    with pytest.raises(ValueError):
        fit.predict(np.ones((10, 2)))


def test_inner_optimizer_failure_is_reported(monkeypatch):
    from types import SimpleNamespace

    from robustmodels import regression

    monkeypatch.setattr(
        regression.optimize,
        "least_squares",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, message="evaluation budget exhausted"
        ),
    )
    X = np.arange(10.0)[:, None]
    with pytest.raises(RuntimeError, match="nonlinear least squares failed: evaluation budget"):
        NonlinearMRegressor(decay, [1, 1]).fit(X, np.ones(10))
