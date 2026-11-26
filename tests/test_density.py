"""Kernel density estimation: Hermite identities, exact integrals, equivariance and frozen R values.

Reference values: tests/data/r_density.json (R's ks; its "meta" entry records the versions and
options). ks's selectors stop their BFGS search earlier: our criteria are lower at our matrices.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss, hermeval
from scipy import integrate, stats
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels import density
from robustmodels.density import (
    KernelDensity,
    kde_multiscale_test,
    kde_permutation_test,
    kde_test,
    plugin_bandwidth,
    scv_bandwidth,
)

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
    assert value(inverse @ H @ inverse.T) <= value(inverse @ reference @ inverse.T) * (1 + 1e-12)


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


def test_kde_permutation_by_quadrature_and_exhaustive_assignments():
    x = np.array([-2.0, -1.0, -1.0, 0.5, 0.5, 2.0, 3.0])
    h = 0.5

    def distance(indices):
        keep = np.isin(np.arange(len(x)), indices)

        def integrand(t):
            pdf = stats.norm.pdf(x, loc=t, scale=h)
            return (pdf[keep].mean() - pdf[~keep].mean()) ** 2

        return integrate.quad(integrand, -10, 10, epsabs=1e-11)[0]

    observed = distance((0, 1, 2))
    null = [distance(i) for i in itertools.combinations(range(len(x)), 3)]
    expected = np.mean(np.asarray(null) >= observed - 1e-11)
    result = kde_permutation_test(x[:3], x[3:], bandwidth=h, n_resamples=np.inf)
    assert result.statistic == pytest.approx(observed, rel=1e-10)
    assert result.pvalue == pytest.approx(expected)


def test_kde_permutation_calibration_and_bandwidth():
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="degenerate normal calibration"):
        kde_test(x, x)
    assert kde_permutation_test(x, x) == (0, 1)
    with pytest.raises(ValueError, match="singular"):
        kde_permutation_test(np.ones(4), np.ones(4))
    assert kde_permutation_test(np.ones(4), np.ones(4), bandwidth=1.0) == (0, 1)
    result = kde_permutation_test(x, x + 2, n_resamples=17, random_state=1)
    scaled = kde_permutation_test(3 * x, 3 * (x + 2), n_resamples=17, random_state=1)
    assert scaled.statistic == pytest.approx(result.statistic / 3)
    assert scaled.pvalue == result.pvalue
    assert result.pvalue * 18 == pytest.approx(round(result.pvalue * 18))
    with pytest.raises(ValueError, match="n_resamples"):
        kde_permutation_test(x, x, n_resamples=0)


def _multiscale_reference(X, H, scales, assignments):
    """Compute every discrepancy from Gaussian densities and group sums."""
    offsets = X[:, None] - X[None, :]
    columns = []
    for scale in scales:
        kernel = stats.multivariate_normal.pdf(offsets, cov=2 * scale**2 * H)
        discrepancies = []
        for a in assignments:
            b = np.setdiff1d(np.arange(len(X)), a)
            within_a = kernel[np.ix_(a, a)].mean()
            within_b = kernel[np.ix_(b, b)].mean()
            between = kernel[np.ix_(a, b)].mean()
            discrepancies.append(within_a + within_b - 2 * between)
        columns.append(discrepancies)
    discrepancies = np.column_stack(columns)
    standardized = (discrepancies - discrepancies.mean(axis=0)) / discrepancies.std(axis=0)
    maximum = standardized.max(axis=1)
    return maximum[0], np.mean(maximum >= maximum[0] - 1e-12)


@pytest.mark.parametrize("budget", [7, 35, np.inf])
def test_kde_multiscale_matches_joint_permutation_reference(budget):
    X = np.array([[-2.0, 0], [-1, 2], [-1, 2], [0.5, -1], [0.5, 3], [2, -2], [3, 1]])
    H = np.array([[0.7, 0.3], [0.3, 1.5]])
    scales = [0.5, 1, 2]
    n1, seed = 3, 17
    if budget >= 35:
        assignments = list(itertools.combinations(range(len(X)), n1))
    else:
        rng = np.random.default_rng(seed)
        assignments = [np.arange(n1)] + [rng.permutation(len(X))[:n1] for _ in range(budget)]
    expected = _multiscale_reference(X, H, scales, assignments)
    result = kde_multiscale_test(
        X[:n1], X[n1:], bandwidth=H, scales=scales, n_resamples=budget, random_state=seed
    )
    assert result.statistic == pytest.approx(expected[0], rel=1e-12)
    assert result.pvalue == expected[1]


def test_kde_multiscale_exact_level_over_assignments_with_ties():
    X = np.array([-2.0, -1.0, -1.0, 0.5, 0.5, 2.0, 3.0])
    pvalues = []
    for group in itertools.combinations(range(len(X)), 3):
        inside = np.isin(np.arange(len(X)), group)
        pvalues.append(kde_multiscale_test(X[inside], X[~inside], n_resamples=np.inf).pvalue)
    pvalues = np.array(pvalues)
    assert np.min(pvalues) >= 1 / len(pvalues)
    for alpha in np.unique(pvalues):
        assert (pvalues <= alpha).mean() <= alpha


@pytest.mark.parametrize("budget", [17, 145, np.inf])
def test_kde_multiscale_one_band_matches_single_band_pvalue_and_rng(budget):
    x = np.array([-2.0, -1.0, 0.0, 1.0])
    y = np.array([0.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    rng_single = np.random.default_rng(16)
    rng_multi = np.random.default_rng(16)
    single = kde_permutation_test(x, y, n_resamples=budget, random_state=rng_single)
    multi = kde_multiscale_test(x, y, scales=[1], n_resamples=budget, random_state=rng_multi)
    assert single.pvalue == multi.pvalue
    rng_many = np.random.default_rng(16)
    many = kde_multiscale_test(x, y, n_resamples=budget, random_state=rng_many)
    assert kde_multiscale_test(x, y, n_resamples=budget, random_state=16) == many
    assert rng_single.random() == rng_multi.random() == rng_many.random()


def test_kde_multiscale_affine_invariance_and_scale_order():
    rng = np.random.default_rng(26)
    x = rng.normal(size=(9, 2))
    y = rng.normal(size=(11, 2)) * [1, 3]
    transform = np.array([[3.0, 0.5], [-1.0, 0.7]])
    base = kde_multiscale_test(x, y, scales=[0.25, 1, 4], n_resamples=79, random_state=8)
    changed = kde_multiscale_test(
        x @ transform + 10, y @ transform + 10, scales=[4, 1, 0.25], n_resamples=79, random_state=8
    )
    assert changed.statistic == pytest.approx(base.statistic, rel=1e-12)
    assert changed.pvalue == base.pvalue


def test_kde_multiscale_omits_constant_scales():
    assert kde_multiscale_test(np.ones(3), np.ones(4), bandwidth=1) == (0, 1)
    x, y = [-3, -2, -1], [0, 1, 2, 3]
    wide = 1e300  # all kernel entries become 1 at this scale
    assert kde_multiscale_test(x, y, scales=[1, wide], n_resamples=np.inf) == kde_multiscale_test(
        x, y, scales=[1], n_resamples=np.inf
    )


@pytest.mark.parametrize("scales", [[], 1, [[1]], [0], [-1], [np.inf], [np.nan]])
def test_kde_multiscale_checks_scales(scales):
    with pytest.raises(ValueError, match="scales"):
        kde_multiscale_test([0, 1], [2, 3], scales=scales)


def test_kernel_derivatives_when_the_determinant_underflows():
    d, h = 10, 1e-20
    kde = KernelDensity(h).fit(np.zeros((2, d)))
    point = np.zeros((1, d))
    pdf = (h * np.sqrt(2 * np.pi)) ** -d
    np.testing.assert_allclose(kde.gradient(point), 0)
    np.testing.assert_allclose(kde.hessian(point)[0] / pdf * h**2, -np.eye(d), rtol=1e-12)


def test_input_checks():
    with pytest.raises(ValueError, match="bandwidth must be one of"):
        KernelDensity("silverman").fit(X2)
    with pytest.raises(ValueError, match="positive"):
        KernelDensity(-1.0).fit(X2)
    with pytest.raises(ValueError, match="positive definite"):
        KernelDensity([[1.0, 2.0], [2.0, 1.0]]).fit(X2)
    with pytest.raises(ValueError, match="symmetric"):
        KernelDensity(np.array([[1, 0], [0.5, 1]]) * 1e-12).fit(X2)
    with pytest.raises(ValueError, match="finite"):
        KernelDensity([[1, 0], [0, np.inf]]).fit(X2)
    for h in (1e200, 1e-200):
        with pytest.raises(ValueError, match="squared bandwidth"):
            KernelDensity(h).fit(X2)
    for order in (1.0, True):
        with pytest.raises(ValueError, match="deriv_order"):
            plugin_bandwidth(X2, deriv_order=order)
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


def test_selectors_accept_columns_of_disparate_scales():
    D = np.diag([1e7, 1e-7])
    for select in (plugin_bandwidth, scv_bandwidth):
        np.testing.assert_allclose(select(X2 @ D), D @ select(X2) @ D, rtol=1e-5)
