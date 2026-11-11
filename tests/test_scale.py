import numpy as np
import pytest
from scipy import integrate, optimize, stats

import reference as ref
from robustmodels import scale
from robustmodels.scale import (
    MAD_CONSTANT,
    PN_CONSTANT,
    QN_CONSTANT,
    SN_CONSTANT,
    lmc,
    m_scale,
    mad,
    medcouple,
    pn,
    qn,
    rmc,
    sn,
    tau_scale,
)


def test_consistency_constants():
    assert MAD_CONSTANT == pytest.approx(1.4826022185)
    assert QN_CONSTANT == pytest.approx(2.2191444660)
    assert PN_CONSTANT == pytest.approx(1.0483580825)

    # Sn: g(x) = med|x - Y| solves Phi(x + g) - Phi(x - g) = 1/2; Sn -> c * g(Phi^-1(3/4)).
    x = stats.norm.ppf(0.75)
    g = optimize.brentq(lambda t: stats.norm.cdf(x + t) - stats.norm.cdf(x - t) - 0.5, 1e-9, 9)
    assert SN_CONSTANT == pytest.approx(1 / g, rel=1e-12)


def test_raw_estimators_match_definitions(samples):
    for x in samples:
        assert qn(x, consistency=False, finite_correction=False) == ref.qn_raw(x)
        assert sn(x, consistency=False, finite_correction=False) == ref.sn_raw(x)
        assert pn(x, consistency=False, finite_correction=False) == ref.pn_raw(x)
        assert medcouple(x) == pytest.approx(ref.medcouple(x), abs=1e-12)


@pytest.mark.parametrize("n", [60, 101, 250])
def test_selection_algorithm_matches_brute_force(monkeypatch, n):
    monkeypatch.setattr(scale, "_DIRECT_LIMIT", 0)  # force the O(n log n) path
    rng = np.random.default_rng(n)
    for x in (rng.normal(size=n), rng.integers(0, 8, size=n).astype(float)):
        assert qn(x, consistency=False, finite_correction=False) == ref.qn_raw(x)
        assert pn(x, consistency=False, finite_correction=False) == ref.pn_raw(x)
        assert medcouple(x) == pytest.approx(ref.medcouple(x), abs=1e-12)


@pytest.mark.parametrize("n", [5, 6, 10, 11])
def test_finite_sample_corrections_remove_bias(n):
    X = np.random.default_rng(n).normal(size=(4000, n))
    for estimator in (qn, sn, pn, lambda x: mad(x, finite_correction=True)):
        assert np.mean([estimator(x) for x in X]) == pytest.approx(1, abs=0.025)


def test_consistent_at_the_normal_model():
    x = np.random.default_rng(0).normal(scale=2, size=20_000)
    for estimator in (mad, qn, sn, pn, tau_scale, m_scale):
        assert estimator(x) == pytest.approx(2, rel=0.03)


def test_equivariance():
    x = np.random.default_rng(1).lognormal(size=37)
    for estimator in (mad, qn, sn, pn, tau_scale, m_scale):
        assert estimator(-3 * x + 7) == pytest.approx(3 * estimator(x), rel=1e-12)
    assert medcouple(-x) == -medcouple(x)
    assert lmc(-x) == pytest.approx(rmc(x), abs=1e-15)


def test_medcouple_special_cases():
    assert medcouple([2.0, 2.0, 2.0]) == 0.0
    assert medcouple(np.arange(10.0)) == 0.0
    assert medcouple(np.random.default_rng(2).exponential(size=500)) > 0.2


def test_small_and_invalid_inputs():
    for estimator in (mad, qn, sn, pn, tau_scale, m_scale):
        assert np.isnan(estimator([]))
        assert estimator([3.0]) == 0.0
        with pytest.raises(ValueError):
            estimator([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        qn(np.ones((3, 2)))


def test_m_scale_solves_its_equation():
    x = np.random.default_rng(3).standard_t(3, size=200)
    s = m_scale(x, center=0)
    c = scale._tuning_for_breakdown(0.5)
    assert np.mean(scale._rho(x / s, c)) == pytest.approx(0.5, abs=1e-8)


def test_m_scale_with_a_zero_initial_mad():
    x = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    breakdown = 0.1
    c = scale._tuning_for_breakdown(breakdown)
    expected = optimize.brentq(lambda s: np.mean(scale._rho(x / s, c)) - breakdown, 0.01, 10)
    assert m_scale(x, center=0, breakdown=breakdown) == pytest.approx(expected, rel=1e-8)
    assert m_scale(x, center=0, breakdown=0.5) == 0


def test_bisquare_constants():
    # closed-form E[rho] against numerical integration, and witnesses from rrcov
    c = 2.3
    inside = integrate.quad(lambda z: scale._rho(z, c) * stats.norm.pdf(z), -c, c, epsabs=1e-14)[0]
    assert scale._expected_rho(c, 1) == pytest.approx(inside + 2 * stats.norm.sf(c), rel=1e-12)
    for p, c_s, c_mm in [(1, 1.54764498, 5.49024921), (2, 2.66080339, 5.81031562),
                         (5, 4.65202334, 6.59558039)]:  # fmt: skip
        assert scale._tuning_for_breakdown(0.5, p) == pytest.approx(c_s, rel=1e-8)
        assert scale._tuning_for_shape_efficiency(0.95, p) == pytest.approx(c_mm, rel=1e-7)


@pytest.mark.parametrize("parameter", ["c1", "c2"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_tau_tuning(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        tau_scale([1, 2, 3], **{parameter: value})


@pytest.mark.parametrize("estimator", [mad, m_scale])
@pytest.mark.parametrize("center", [np.nan, np.inf, -np.inf])
def test_nonfinite_center_is_rejected(estimator, center):
    with pytest.raises(ValueError, match="center"):
        estimator([1, 2, 3], center=center)


@pytest.mark.parametrize("estimator", [mad, qn, sn, pn, tau_scale, m_scale, medcouple, lmc, rmc])
def test_scale_does_not_modify_input(estimator):
    x = np.array([9.0, 0.0, 4.0, 2.0, 1.0])
    original = x.copy()
    estimator(x)
    np.testing.assert_array_equal(x, original)


def test_tau_scale_with_no_surviving_location_weights():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    raw_mad = 1.0
    expected = raw_mad * np.sqrt(np.mean(((x - np.median(x)) / raw_mad) ** 2))
    with np.errstate(all="raise"):
        assert tau_scale(x, c1=0.01, consistency=False) == pytest.approx(expected)


def test_tuning_exists_for_small_breakdown_and_many_variables():
    for breakdown, p in [(1e-6, 1), (1e-4, 200)]:
        c = scale._tuning_for_breakdown(breakdown, p)
        assert scale._expected_rho(c, p) == pytest.approx(breakdown, rel=1e-6)


def test_non_numeric_parameters_raise_value_error():
    with pytest.raises(ValueError, match="center"):
        mad([1, 2, 3], center="a")
    with pytest.raises(ValueError, match="breakdown"):
        m_scale([1, 2, 3], breakdown=None)
    assert str(lmc([1.0])) == "0.0"


@pytest.mark.parametrize("breakdown", [0, 0.6, np.nan, np.inf])
def test_invalid_m_scale_breakdown(breakdown):
    with pytest.raises(ValueError, match="breakdown"):
        m_scale([1, 2, 3], breakdown=breakdown)


@pytest.mark.parametrize(
    "n, factor",
    [
        (3, 1.128),
        (4, 1.303),
        (5, 1.109),
        (39, 1.020),
        (40, 1.018),
        (41, 41 / 40.3),
        (100, 100 / 99.3),
    ],
)
def test_pn_finite_correction_matches_tarr(n, factor):
    x = np.random.default_rng(n).normal(size=n)
    raw = ref.pn_raw(x)
    expected = raw * PN_CONSTANT * factor
    assert pn(x) == pytest.approx(expected)
    assert pn(x, consistency=False) == pytest.approx(raw * factor)
    assert pn(x, finite_correction=False) == pytest.approx(raw * PN_CONSTANT)


@pytest.mark.parametrize("n", [3, 4])
def test_pn_small_sample_factors_against_normal_order_statistics(n):
    # For n=3 the raw pairwise IQR has expectation E[X_(3)]. For n=4 it
    # has expectation E[X_(4)] + E[X_(2)] (Tarr, Appendix B.1).
    expected_max = integrate.quad(
        lambda x: x * n * stats.norm.cdf(x) ** (n - 1) * stats.norm.pdf(x),
        -np.inf,
        np.inf,
    )[0]
    expected_raw = expected_max
    if n == 4:
        expected_raw += integrate.quad(
            lambda x: x * 12 * stats.norm.cdf(x) * stats.norm.sf(x) ** 2 * stats.norm.pdf(x),
            -np.inf,
            np.inf,
        )[0]
    x = np.arange(float(n))
    applied_factor = pn(x) / pn(x, finite_correction=False)
    assert expected_raw * PN_CONSTANT * applied_factor == pytest.approx(1, abs=0.002)


def test_pn_two_observations_have_no_pairwise_spread():
    assert pn([1, 9]) == 0
    assert pn([1, 9], finite_correction=False) == 0


@pytest.mark.parametrize("breakdown,n", [(0.5, 101), (0.49, 101), (0.1, 10)])
def test_m_scale_finishes_slow_fixed_point_with_a_bracketed_root(breakdown, n):
    # Close to exact fit or spanning six orders of magnitude. The former
    # 200-step iteration silently returned residuals as large as 5.5e-4.
    from scipy.optimize import brentq

    x = np.r_[np.zeros(n // 2), np.geomspace(1, 1e6, n - n // 2)]
    c = scale._tuning_for_breakdown(breakdown)

    def equation(s):
        u = np.minimum(np.abs(x / (s * c)), 1)
        return np.mean(3 * u**2 - 3 * u**4 + u**6) - breakdown

    oracle = brentq(equation, 1e-6, 1e8, xtol=1e-12)
    fitted = scale.m_scale(x, center=0, breakdown=breakdown)
    assert fitted == pytest.approx(oracle, rel=1e-8)
    assert abs(equation(fitted)) < 1e-10


def test_medcouple_tie_kernel_uses_average_of_middle_values():
    # Brys et al., equations (2.1)-(2.3): kernels for [-3,0,0] are
    # [-1,-1,-1,0,0,1]. R robustbase 0.99-7 returns -1 in this case.
    assert medcouple([-3.0, 0.0, 0.0]) == -0.5
    assert medcouple([3.0, 0.0, 0.0]) == 0.5


def test_skewness_and_tail_weight_of_an_empty_sample_are_nan():
    for estimator in (medcouple, lmc, rmc):
        assert np.isnan(estimator([]))


def test_tau_scale_is_zero_when_more_than_half_the_values_are_tied():
    assert tau_scale([3.0, 3.0, 3.0, 3.0, 10.0]) == 0.0
