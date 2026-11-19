"""Small exact definitions, affine equivariance, degeneracies and frozen R values."""

import json
from pathlib import Path

import numpy as np
import pytest

from robustmodels.depth import bagplot_stats, halfspace_depth, halfspace_median, regression_depth

R = json.loads((Path(__file__).parent / "data" / "r_depth.json").read_text())


def brute_halfplanes(X, z):
    differences = X - z
    offsets = differences[np.any(differences != 0, axis=1)]
    if not len(offsets):
        return 1.0
    angles = np.arctan2(offsets[:, 1], offsets[:, 0]) + np.pi / 2
    # Perturb both sides of every critical direction to enumerate open angular cells.
    angles = np.r_[angles + 1e-7, angles - 1e-7, angles + np.pi + 1e-7, angles + np.pi - 1e-7]
    normals = np.c_[np.cos(angles), np.sin(angles)]
    return np.count_nonzero(differences @ normals.T >= 0, axis=0).min() / len(X)


def test_bivariate_depth_by_halfplane_enumeration():
    rng = np.random.default_rng(0)
    for trial in range(25):
        X = rng.normal(size=(12, 2)) if trial % 2 else rng.integers(0, 4, size=(12, 2))
        points = np.vstack([X, rng.normal(size=(3, 2))])
        expected = [brute_halfplanes(X, z) for z in points]
        np.testing.assert_allclose(halfspace_depth(X, points), expected)
    np.testing.assert_array_equal(halfspace_depth(np.ones((6, 2))), 1)


def test_univariate_depth_including_ties_and_external_points():
    X = np.array([0, 0, 1, 2, 2])[:, None]
    points = np.array([-1, 0, 0.5, 1, 2, 3])[:, None]
    expected = [min(np.mean(X <= p), np.mean(X >= p)) for p in points]
    np.testing.assert_allclose(halfspace_depth(X, points), expected)


@pytest.mark.parametrize("name", R["hdepth"])
def test_halfspace_r_reference(name):
    r = R["hdepth"][name]
    np.testing.assert_allclose(
        halfspace_depth(r["x"], r["z"], n_directions=10000, random_state=0), r["depth"], atol=1e-12
    )


@pytest.mark.parametrize("name", ["stars", "p2"])
def test_regression_r_reference(name):
    r = R["rdepth"][name]
    xy = np.array(r["x"])
    for coef, expected in zip(r["fits"], r["depth"], strict=True):
        assert regression_depth(
            xy[:, :-1], xy[:, -1], coef[1:], coef[0], n_directions=10000, random_state=0
        ) == pytest.approx(expected)


def test_regression_depth_by_enumerating_univariate_cuts():
    rng = np.random.default_rng(1)
    for _ in range(30):
        x, residuals = rng.integers(0, 6, size=(2, 15)) - 2
        unique = np.unique(x)
        cuts = np.r_[unique[0] - 1, (unique[1:] + unique[:-1]) / 2, unique[-1] + 1]
        expected = len(x)
        for cut in cuts:
            count = np.sum((x < cut) & (residuals >= 0)) + np.sum((x > cut) & (residuals <= 0))
            other = np.sum((x < cut) & (residuals <= 0)) + np.sum((x > cut) & (residuals >= 0))
            expected = min(expected, count, other)
        assert regression_depth(x[:, None], residuals, [0]) == expected / len(x)


@pytest.mark.parametrize("name", ["normal2", "stars", "skew2"])
def test_tukey_median_against_r(name):
    r = R["bagplot"][name]
    # R uses iterative numerical tolerances for degenerate deepest regions.
    np.testing.assert_allclose(halfspace_median(r["x"]), np.ravel(r["center"]), atol=2e-5, rtol=0)


def test_bag_fence_loop_and_affine_transform():
    X = np.random.default_rng(0).normal(size=(40, 2))
    X[:2] = [[6, 6], [-7, 6]]
    fit = bagplot_stats(X)
    np.testing.assert_allclose(fit.fence, fit.center + 3 * (fit.bag - fit.center))
    assert fit.outliers[:2].all()
    np.testing.assert_allclose(fit.depth, halfspace_depth(X))
    A, shift = np.array([[2.0, 1], [-0.5, 3]]), np.array([8.0, -4])
    transformed = bagplot_stats(X @ A + shift)
    np.testing.assert_allclose(transformed.center, fit.center @ A + shift, atol=1e-7)
    np.testing.assert_array_equal(transformed.outliers, fit.outliers)
    np.testing.assert_allclose(halfspace_depth(X @ A + shift), fit.depth)
    # Polygon vertices can differ under a finite angular interpolation grid.
    for points in [fit.bag, X[~fit.outliers]]:
        edges = np.roll(fit.loop, -1, axis=0) - fit.loop
        delta = points[:, None] - fit.loop
        cross = edges[:, 0] * delta[:, :, 1] - edges[:, 1] * delta[:, :, 0]
        assert cross.min() >= -1e-8


@pytest.mark.parametrize(
    "X, center",
    [
        ([[0, 0], [1, 0], [0, 1]], [1 / 3, 1 / 3]),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], [0.5, 0.5]),
        ([[0, 0]] * 6 + [[1, 0], [0, 1]], [0, 0]),
    ],
)
def test_degenerate_deepest_regions(X, center):
    fit = bagplot_stats(X)
    np.testing.assert_allclose(fit.center, center, atol=1e-9)
    for vertices in [fit.bag, fit.fence, fit.loop]:
        assert np.isfinite(vertices).all()
    if center == [0, 0]:
        np.testing.assert_array_equal(fit.outliers, [False] * 6 + [True, True])


def test_seeded_subsampling_and_validation():
    X = np.random.default_rng(7).normal(size=(35, 2))
    a = bagplot_stats(X, max_samples=20, random_state=0)
    b = bagplot_stats(X, max_samples=20, random_state=0)
    np.testing.assert_array_equal(a.bag, b.bag)
    for max_samples in [1, 2, 0]:
        with pytest.raises(ValueError, match="max_samples"):
            halfspace_median(X, max_samples=max_samples)
    with pytest.raises(ValueError, match="bivariate"):
        halfspace_median(np.ones((10, 3)))
    with pytest.raises(ValueError, match="singular"):
        bagplot_stats(np.ones((10, 2)))
    with pytest.raises(ValueError, match="factor"):
        bagplot_stats(X, factor=0.5)
    with pytest.raises(ValueError, match="features"):
        halfspace_depth(X, [[1, 2, 3]])
    with pytest.raises(ValueError, match="n_directions"):
        halfspace_depth(X, n_directions=0)
    with pytest.raises(ValueError, match="entries"):
        regression_depth(X, np.arange(35), [0])
    with pytest.raises(ValueError, match="finite"):
        regression_depth(X, np.arange(35), [np.nan, 0])
    with pytest.raises(ValueError, match="n_directions"):
        regression_depth(X, np.arange(35), [0, 0], n_directions=0)


def test_regression_depth_in_three_dimensions_by_exhaustive_separation():
    import itertools

    from scipy.optimize import linprog

    rng = np.random.default_rng(6)
    X = rng.normal(size=(8, 3))
    design = np.c_[np.ones(8), X]
    for _ in range(5):
        y = rng.normal(size=8)
        coef = rng.normal(size=4)
        signs = np.sign(y - design @ coef)
        # A nonfit has its positive and negative residuals strictly separated
        # by an affine hyperplane in X. Enumerate every possible deletion.
        for removed in range(9):
            feasible = False
            for kept in itertools.combinations(range(8), 8 - removed):
                signed = design[list(kept)] * signs[list(kept), None]
                fit = linprog(
                    np.zeros(4), A_ub=-signed, b_ub=-np.ones(len(kept)), bounds=[(None, None)] * 4
                )
                if fit.success:
                    feasible = True
                    break
            if feasible:
                break
        assert regression_depth(X, y, coef[1:], coef[0], n_directions=1000) == removed / 8


def test_car_median_has_greater_depth_than_r_approximation():
    r = R["bagplot"]["cars"]
    X = np.array(r["x"])
    ours = halfspace_median(X)
    np.testing.assert_allclose(halfspace_depth(X, [ours]), [25 / 60])
    np.testing.assert_allclose(halfspace_depth(X, [r["center"]]), [24 / 60])


def test_singular_selected_subset_has_actionable_error():
    X = np.vstack([np.zeros((18, 2)), [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="selected subset.*increase max_samples"):
        halfspace_median(X, max_samples=3, random_state=0)


def test_univariate_closed_halfspaces_do_not_merge_distinct_query_points():
    X = np.array([[0.0], [1.0]])
    points = np.array([[-1e-12], [0.0], [1.0], [1.0 + 1e-12]])
    np.testing.assert_array_equal(halfspace_depth(X, points), [0, 0.5, 0.5, 0])


def test_bag_interpolates_a_nonzero_inner_segment():
    from robustmodels.depth import _radius, _Regions

    # The depth-2 region is [(1, 1), (2, 2)], with two data points;
    # the depth-1 region has all five. The ray to (3, 3) therefore has
    # interpolation coefficient (2.5 - 2)/(5 - 2) = 1/6 on the outer region.
    X = np.array([[1.0, 1], [1, 1], [1, 2], [3, 2], [3, 3]])
    regions = _Regions(X, 200, None)
    center = regions.median()
    stats = bagplot_stats(X)
    # Along the diagonal the bag boundary is 1/6*(3,3) + 5/6*(2,2).
    target = np.array([13 / 6, 13 / 6])
    angle = np.arctan2(*(target - center)[::-1])
    radial = _radius(stats.bag, center, [angle])[0]
    np.testing.assert_allclose(
        center + radial * np.array([np.cos(angle), np.sin(angle)]), target, atol=1e-8
    )


def test_narrow_bivariate_halfplanes_against_exact_rational_signs():
    from fractions import Fraction

    X = np.array([[1.0, 0], [1, 2e-14], [-1, 0], [0, -2e-14], [0, 1e-14], [1, 2e-14], [-1, 2e-14]])
    vectors = [[Fraction(float(v)) for v in row] for row in X]
    best = len(X)
    # Enumerate infinitesimal rotations of the normal to each observation.
    # Exact rational signs determine the count on each open angular cell;
    # this calculation uses neither arctan2 nor a numerical perturbation.
    for x, y in vectors:
        for sign in [-1, 1]:
            for side in [-1, 1]:
                count = 0
                for a, b in vectors:
                    leading = sign * (x * b - y * a)
                    perturbation = side * (x * a + y * b)
                    count += leading > 0 or (leading == 0 and perturbation >= 0)
                best = min(best, count)
    assert best == 2
    assert halfspace_depth(X, [[0.0, 0.0]])[0] == best / len(X)
    # Halfspace depth is invariant under nonsingular coordinate scaling.
    assert halfspace_depth(X * [1, 1e14], [[0.0, 0.0]])[0] == best / len(X)


def test_bivariate_depth_at_midpoints_by_exact_rational_signs():
    """Opposite offsets (the query between two observations) must not open a spurious cell."""
    from fractions import Fraction

    def exact(X, z):
        offsets = [(Fraction(a) - Fraction(z[0]), Fraction(b) - Fraction(z[1])) for a, b in X]
        away = [v for v in offsets if v != (0, 0)]
        best = len(away)
        for a, b in away:  # normals of each offset, rotated infinitesimally either way
            for sign in (1, -1):
                for t in (Fraction(1, 10**12), -Fraction(1, 10**12)):
                    u = (-b * sign + t * a, a * sign + t * b)
                    best = min(best, sum(x * u[0] + y * u[1] >= 0 for x, y in away))
        return (len(X) - len(away) + best) / len(X)

    rng = np.random.default_rng(11)
    for _ in range(300):
        X = rng.integers(-3, 4, size=(9, 2)) * rng.choice([1, 0.5, 0.25])
        i, j = rng.choice(9, 2, replace=False)
        z = (X[i] + X[j]) / 2  # exact in binary
        assert halfspace_depth(X, [z])[0] == exact(X, z)
    # A case that 0.2.0 answered with 2/9.
    X = [[-1, 0], [2, 2], [1, 2], [3, -1], [-2, 2], [-2, 3], [2, -1], [2, 3], [-2, -2]]
    assert halfspace_depth(X, [[0.5, 1.5]])[0] == pytest.approx(3 / 9)


def lp_depth(X, z):
    """``n`` minus the largest set strictly on one side of a hyperplane through ``z``."""
    from itertools import combinations

    from scipy.optimize import linprog

    A = np.asarray(X, float) - z
    n, d = A.shape
    candidates = np.flatnonzero(A.any(axis=1))
    for k in range(len(candidates), 0, -1):
        for subset in combinations(candidates, k):
            fit = linprog(np.zeros(d), A[list(subset)], -np.ones(k), bounds=[(None, None)] * d)
            if fit.status == 0:
                return (n - k) / n
    return 1.0


@pytest.mark.parametrize("p", [3, 4])
def test_exact_depth_by_linear_programming_separation(p):
    rng = np.random.default_rng(p)
    for trial in range(12):
        X = rng.integers(-2, 3, size=(8, p)).astype(float) if trial % 2 else rng.normal(size=(8, p))
        points = np.vstack([X[:2], X.mean(axis=0), rng.normal(size=(1, p)), [3 * np.ones(p)]])
        expected = [lp_depth(X, z) for z in points]
        np.testing.assert_allclose(halfspace_depth(X, points, exact=True), expected)


def test_exact_depth_bounds_and_invariance():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(25, 3))
    exact = halfspace_depth(X, exact=True)
    approximate = halfspace_depth(X, n_directions=300, random_state=0)
    assert np.all(approximate >= exact)
    A, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    np.testing.assert_array_equal(halfspace_depth(X @ A.T + b, exact=True), exact)
    # Data on a plane: the depth is the bivariate depth within the plane.
    plane = np.c_[X[:, :2], np.zeros(25)] @ A.T
    np.testing.assert_array_equal(halfspace_depth(plane, exact=True), halfspace_depth(X[:, :2]))
    # Data on a line: the univariate depth along it.
    line = np.outer(np.r_[-2.0, -1, 0, 1, 3], [1.0, 2.0, -1.0])
    np.testing.assert_array_equal(halfspace_depth(line, exact=True), [0.2, 0.4, 0.6, 0.4, 0.2])
    # Ties and points that coincide with the query.
    tied = np.repeat(X[:5], 3, axis=0)
    assert halfspace_depth(tied, tied[:1], exact=True)[0] == pytest.approx(lp_depth(tied, tied[0]))


@pytest.mark.parametrize("name", R["hdepth_exact"])
def test_exact_depth_ddalpha_reference(name):
    r = R["hdepth_exact"][name]
    np.testing.assert_allclose(halfspace_depth(r["x"], r["z"], exact=True), r["depth"], atol=1e-12)
