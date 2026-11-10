import numpy as np
import pytest
from scipy import stats

import reference as ref
from robustmodels import scale
from robustmodels.location import (
    hodges_lehmann,
    m_location,
    tau_location,
    trimmed_mean,
    winsorized_mean,
)
from robustmodels.scale import mad

ESTIMATORS = (hodges_lehmann, m_location, tau_location, trimmed_mean, winsorized_mean)


def test_hodges_lehmann_matches_definition(samples, monkeypatch):
    for x in samples:
        assert hodges_lehmann(x) == ref.hodges_lehmann(x)
    monkeypatch.setattr(scale, "_DIRECT_LIMIT", 0)
    x = np.random.default_rng(0).normal(size=301)
    assert hodges_lehmann(x) == ref.hodges_lehmann(x)


def test_trimmed_and_winsorized_means_match_scipy():
    x = np.random.default_rng(1).standard_t(2, size=57)
    assert trimmed_mean(x, 0.2) == pytest.approx(stats.trim_mean(x, 0.2))
    expected = np.mean(stats.mstats.winsorize(x, limits=(0.2, 0.2)))
    assert winsorized_mean(x, 0.2) == pytest.approx(expected)


@pytest.mark.parametrize("psi", ["huber", "bisquare"])
def test_m_location_solves_its_equation(psi):
    x = np.random.default_rng(2).standard_t(3, size=200)
    mu, s = m_location(x, psi=psi), mad(x)
    r = (x - mu) / s
    if psi == "huber":
        score = np.clip(r, -1.345, 1.345)
    else:
        score = r * (1 - (r / 4.685) ** 2) ** 2 * (np.abs(r) < 4.685)
    assert np.sum(score) == pytest.approx(0, abs=1e-6)


def test_equivariance_and_robustness():
    rng = np.random.default_rng(3)
    x = rng.normal(size=101)
    for estimator in ESTIMATORS:
        assert estimator(-2 * x + 5) == pytest.approx(-2 * estimator(x) + 5)
    x[:15] = 1000  # 15% gross outliers
    for estimator in (hodges_lehmann, m_location, tau_location, lambda z: trimmed_mean(z, 0.2)):
        assert abs(estimator(x)) < 1


def test_small_and_invalid_inputs():
    for estimator in ESTIMATORS:
        assert np.isnan(estimator([]))
        assert estimator([3.0]) == 3.0
        with pytest.raises(ValueError):
            estimator([1.0, np.inf])
    with pytest.raises(ValueError):
        trimmed_mean([1.0, 2.0], proportion=0.5)
    with pytest.raises(ValueError):
        m_location([1.0, 2.0], psi="cauchy")


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"c": 0}, "c"),
        ({"c": np.nan}, "c"),
        ({"scale": -1}, "scale"),
        ({"scale": np.inf}, "scale"),
        ({"tol": 0}, "tol"),
        ({"tol": np.nan}, "tol"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
    ],
)
def test_invalid_m_location_parameters(kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        m_location([1, 2, 3], **kwargs)


def test_bisquare_with_no_surviving_weights_keeps_the_median():
    with np.errstate(all="raise"):
        assert m_location([0, 1, 2, 3], psi="bisquare", c=0.01, scale=1) == 1.5
        assert tau_location([0, 1, 2, 3], c1=0.01) == 1.5


@pytest.mark.parametrize("psi", ["huber", "bisquare"])
def test_zero_scale_returns_the_median(psi):
    assert m_location([1, 2, 9], psi=psi, scale=0) == 2


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_location_does_not_modify_input(estimator):
    x = np.array([9.0, 0.0, 4.0, 2.0, 1.0])
    original = x.copy()
    estimator(x)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_tau_tuning(value):
    with pytest.raises(ValueError, match="c1"):
        tau_location([1, 2, 3], c1=value)


def test_nan_trimming_proportion_is_rejected():
    for estimator in (trimmed_mean, winsorized_mean):
        with pytest.raises(ValueError, match="proportion"):
            estimator([1, 2, 3], proportion=np.nan)


def test_m_location_warns_instead_of_silently_accepting_nonconvergence():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        m_location([1.0, 2.0, 2.0, 3.0, 4.0, 100.0], max_iter=1)
