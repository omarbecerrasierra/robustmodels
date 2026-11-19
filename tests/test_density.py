"""Kernel density estimation: Hermite identities, exact integrals, equivariance and frozen R values.

Reference values: tests/data/r_density.json (R's ks; its "meta" entry records the versions and
options). ks's selectors stop their BFGS search earlier: our criteria are lower at our matrices.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss, hermeval
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels import density
from robustmodels.density import KernelDensity, kde_test, plugin_bandwidth, scv_bandwidth

R = json.loads((Path(__file__).parent / "data" / "r_density.json").read_text())
X2, X3 = np.array(R["x2"]), np.array(R["x3"])


@parametrize_with_checks([KernelDensity(), KernelDensity(0.5)])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_normal_derivative_sums_match_hermite_polynomials():
    # In one dimension the derivative of order 2r of phi_g is g^-2r He_2r(x / g) phi_g(x).
    rng = np.random.default_rng(0)
    x, y, g = rng.normal(size=7), rng.normal(size=5), 0.7
    diff = (x[:, None] - y[None, :]).ravel()
    phi = np.exp(-(diff**2) / (2 * g**2)) / (g * np.sqrt(2 * np.pi))
    for r in range(6):
        hermite = hermeval(diff / g, np.eye(2 * r + 1)[2 * r]) / g ** (2 * r)
        expected = np.mean(phi * hermite)
        got = density._normal_sum(x[:, None], y[:, None], np.array([[g**2]]), r)
        assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("r", [0, 1, 2])
def test_bias_coefficients_by_exact_gaussian_quadrature(r):
    # The coefficients are polynomial in the covariance -a I; check them for a real covariance
    # c I, where Gauss-Hermite quadrature of (u'u)^r (u'Hu)^2 with u ~ N(z, c I) is exact.
    rng = np.random.default_rng(r)
    nodes, weights = hermegauss(8)
    c, z = 0.6, rng.normal(size=2)
    A = rng.normal(size=(2, 2))
    H = A @ A.T
    u1, u2 = np.meshgrid(nodes, nodes, indexing="ij")
    u = z + np.sqrt(c) * np.stack([u1.ravel(), u2.ravel()], axis=1)
    w = np.outer(weights, weights).ravel() / (2 * np.pi)
    quad = np.einsum("ij,jk,ik->i", u, H, u)
    expected = w @ ((u**2).sum(axis=1) ** r * quad**2)
    alpha, beta, gamma, delta, epsilon = density._bias_coefficients(z @ z, -c, 2, r)
    t, v = z @ H @ z, z @ H @ H @ z
    got = alpha * t**2 + beta * np.trace(H) * t + gamma * np.trace(H) ** 2 + delta * v
    assert got + epsilon * np.trace(H @ H) == pytest.approx(expected, rel=1e-12)


def test_estimate_integrates_to_one_with_matching_derivatives():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    kde = KernelDensity([[0.3, 0.1], [0.1, 0.2]]).fit(X)
    axis = np.linspace(-6, 6, 301)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    f = np.exp(kde.score_samples(grid))
    assert f.sum() * (axis[1] - axis[0]) ** 2 == pytest.approx(1, abs=1e-8)
    points, step = rng.normal(size=(5, 2)), 1e-5
    shifts = step * np.eye(2)

    def pdf(points):
        return np.exp(kde.score_samples(points))

    numeric = np.stack([(pdf(points + e) - pdf(points - e)) / (2 * step) for e in shifts], axis=1)
    np.testing.assert_allclose(kde.gradient(points), numeric, rtol=1e-6, atol=1e-10)
    numeric = np.stack(
        [(kde.gradient(points + e) - kde.gradient(points - e)) / (2 * step) for e in shifts], axis=1
    )
    np.testing.assert_allclose(kde.hessian(points), numeric, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("select", [plugin_bandwidth, scv_bandwidth])
def test_selectors_are_affine_equivariant(select):
    A = np.array([[2.0, 0.5], [-1.0, 1.5]])
    for r in (0, 1):
        H = select(X2, deriv_order=r)
        np.testing.assert_allclose(select(X2 @ A.T + 3, deriv_order=r), A @ H @ A.T, rtol=1e-4)


def check_selection(X, select, criterion, r, reference):
    H, reference = select(X, deriv_order=r), np.array(reference)
    np.testing.assert_allclose(H, reference, rtol=1e-3, atol=1e-3 * np.abs(reference).max())
    Z, root = density._sphere(X)
    inverse = np.linalg.inv(root)
    value = criterion(Z, r)
    assert value(inverse @ H @ inverse) <= value(inverse @ reference @ inverse) * (1 + 1e-12)


def test_selectors_match_ks_and_reach_lower_criteria():
    for r in range(3):
        check_selection(X2, plugin_bandwidth, density._plugin_criterion, r, R["hpi"][r])
        check_selection(X2, scv_bandwidth, density._scv_criterion, r, R["hscv"][r])
    check_selection(X3, plugin_bandwidth, density._plugin_criterion, 0, R["hpi3"])
    check_selection(X3, scv_bandwidth, density._scv_criterion, 0, R["hscv3"])


def test_density_derivatives_and_levels_match_ks():
    kde = KernelDensity(R["H"]).fit(X2)
    points = R["points"]
    np.testing.assert_allclose(np.exp(kde.score_samples(points)), R["density"], rtol=1e-12)
    np.testing.assert_allclose(kde.gradient(points), R["gradient"], rtol=1e-12)
    hessian = kde.hessian(points).reshape(len(points), -1)
    np.testing.assert_allclose(hessian, R["hessian"], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(kde.hdr_level([0.25, 0.5, 0.75]), R["levels"], rtol=1e-12)
    train = np.exp(kde.score_samples(X2))
    assert np.mean(train >= kde.hdr_level(0.5)) == pytest.approx(0.5, abs=1 / len(X2))


def test_kde_test_statistic_is_the_integrated_squared_difference():
    rng = np.random.default_rng(2)
    X1, X2_ = rng.normal(size=(30, 2)), rng.normal(size=(25, 2)) + 0.5
    H = np.array([[0.4, 0.1], [0.1, 0.3]])
    T = (
        density._normal_sum(X1, X1, H, 0)
        + density._normal_sum(X2_, X2_, H, 0)
        - 2 * density._normal_sum(X1, X2_, H, 0)
    )
    # phi_H is the convolution of two phi_{H/2}: T is int (f1 - f2)^2 for bandwidth H / 2.
    f1, f2 = KernelDensity(H / 2).fit(X1), KernelDensity(H / 2).fit(X2_)
    axis = np.linspace(-6, 7, 261)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    difference = np.exp(f1.score_samples(grid)) - np.exp(f2.score_samples(grid))
    assert T == pytest.approx(np.sum(difference**2) * (axis[1] - axis[0]) ** 2, rel=1e-8)


def test_kde_test_matches_ks():
    Y1, Y2 = np.array(R["y1"]), np.array(R["y2"])
    np.testing.assert_allclose(density._functional_bandwidth(Y1), R["H1"], rtol=1e-12)
    np.testing.assert_allclose(density._functional_bandwidth(Y2), R["H2"], rtol=1e-12)
    variances = [density._gradient_variance(Y1), density._gradient_variance(Y2)]
    np.testing.assert_allclose(variances, R["var_fhat"], rtol=1e-12)
    result = kde_test(Y1, Y2)
    expected = R["test"]
    assert result.statistic == pytest.approx(expected["statistic"], rel=1e-12)
    assert result.z == pytest.approx(expected["z"], rel=1e-12)
    assert result.pvalue == pytest.approx(expected["pvalue"], rel=1e-10)
    one = kde_test(Y1[:, 0], Y2[:, 0])  # one-dimensional samples
    assert 0 < one.pvalue < 1


def test_input_checks():
    with pytest.raises(ValueError, match="bandwidth must be one of"):
        KernelDensity("silverman").fit(X2)
    with pytest.raises(ValueError, match="positive"):
        KernelDensity(-1.0).fit(X2)
    with pytest.raises(ValueError, match="positive definite"):
        KernelDensity([[1.0, 2.0], [2.0, 1.0]]).fit(X2)
    with pytest.raises(ValueError, match="deriv_order"):
        KernelDensity(deriv_order=3).fit(X2)
    with pytest.raises(ValueError, match="deriv_order"):
        plugin_bandwidth(X2, deriv_order=3)
    with pytest.raises(ValueError, match="deriv_order"):
        scv_bandwidth(X2, deriv_order=-1)
    with pytest.raises(ValueError, match="singular"):
        plugin_bandwidth(np.c_[X2[:, 0], 2 * X2[:, 0]])
    with pytest.raises(ValueError, match="prob"):
        KernelDensity(0.5).fit(X2).hdr_level(1.0)
    with pytest.raises(ValueError, match="variables"):
        kde_test(X2, X3)
    assert density._variance(np.zeros((2, 2)), 10, 0) == np.inf
