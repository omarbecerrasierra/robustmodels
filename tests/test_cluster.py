"""Eigenvalue restriction by brute force, fixed points, recovery and frozen R values.

Reference values: tests/data/r_cluster.json (R's tclust and tkmeans; its
"meta" entry records the versions).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels.cluster import TClust, _restrict_eigenvalues

R = json.loads((Path(__file__).parent / "data" / "r_cluster.json").read_text())


@parametrize_with_checks([TClust(n_init=20, random_state=0)])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def restriction_loss(d, w, m, factor):
    e = np.clip(d, m, factor * m)
    return (w[:, None] * (np.log(e) + d / e)).sum()


def test_eigenvalue_restriction_is_the_optimal_truncation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k, p = rng.integers(1, 4), rng.integers(1, 4)
        d = rng.lognormal(sigma=2, size=(k, p))
        sizes = rng.integers(1, 50, size=k).astype(float)
        factor = float(rng.choice([1, 3, 12, 100]))
        restricted = _restrict_eigenvalues(d, sizes, factor)
        assert restricted.max() <= factor * restricted.min() * (1 + 1e-12)
        w = sizes / sizes.sum()
        m = restricted.min()
        grid = np.geomspace(d.min() / factor / 2, d.max() * 2, 20000)
        best = min(restriction_loss(d, w, g, factor) for g in grid)
        assert restriction_loss(d, w, m, factor) <= best + 1e-9


def test_eigenvalue_restriction_special_cases():
    d = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(_restrict_eigenvalues(d, np.array([5.0, 5.0]), 12), d)
    empty = _restrict_eigenvalues(d, np.array([5.0, 0.0]), 12)  # an empty cluster
    np.testing.assert_array_equal(empty, [[1.0, 2.0], [1.5, 1.5]])
    assert _restrict_eigenvalues(np.zeros((2, 2)), np.array([5.0, 5.0]), 12) is None


def planted(seed=1):
    rng = np.random.default_rng(seed)
    A = rng.multivariate_normal([0, 0], np.eye(2), 100)
    B = rng.multivariate_normal([5, 5], [[4, 0], [0, 0.3]], 80)
    C = rng.multivariate_normal([-4, 6], [[1, 0.8], [0.8, 1]], 60)
    noise = rng.uniform(-10, 12, (12, 2))
    return np.vstack([A, B, C, noise]), np.repeat([0, 1, 2, -1], [100, 80, 60, 12])


def test_recovers_planted_clusters_and_fixed_point():
    X, truth = planted()
    fit = TClust(3, random_state=0).fit(X)
    kept = truth >= 0
    assert adjusted_rand_score(truth[kept], fit.labels_[kept]) > 0.95
    assert np.sum(fit.labels_ == -1) == len(X) - int(np.floor(len(X) * 0.95))
    # The fitted model assigns and trims its own training data the same way.
    np.testing.assert_array_equal(fit.predict(X), fit.labels_)
    np.testing.assert_allclose(fit.weights_.sum(), 1)
    for covariance in fit.covariances_:
        np.testing.assert_allclose(covariance, covariance.T)
    eigenvalues = np.linalg.eigvalsh(fit.covariances_)
    assert eigenvalues.max() <= 12 * eigenvalues.min() * (1 + 1e-10)


def test_trimmed_k_means_assigns_to_the_nearest_center():
    X, _ = planted(2)
    fit = TClust(3, restr_factor=1, equal_weights=True, random_state=0).fit(X)
    distances = ((X[:, None, :] - fit.means_) ** 2).sum(axis=2)
    kept = fit.labels_ >= 0
    np.testing.assert_array_equal(fit.labels_[kept], distances[kept].argmin(axis=1))
    assert distances[~kept].min(axis=1).min() >= distances[kept].min(axis=1).max()
    for covariance in fit.covariances_:
        np.testing.assert_allclose(covariance, covariance[0, 0] * np.eye(2), atol=1e-12)


def same_partition(a, b):
    return adjusted_rand_score(a, b) == 1 and np.array_equal(a == -1, b == -1)


def test_tclust_reference():
    X = np.array(R["x"])
    fit = TClust(3, alpha=0.05, random_state=0).fit(X)
    assert same_partition(fit.labels_, np.array(R["cluster"]) - 1)
    np.testing.assert_allclose(fit.means_, R["centers"], atol=1e-10)
    np.testing.assert_allclose(fit.weights_, R["weights"], atol=1e-12)
    assert fit.objective_ == pytest.approx(R["obj"], rel=1e-12)
    trimmed = TClust(3, alpha=0.05, restr_factor=1, equal_weights=True, random_state=0).fit(X)
    assert same_partition(trimmed.labels_, np.array(R["tk_cluster"]) - 1)
    np.testing.assert_allclose(trimmed.means_, R["tk_centers"], atol=1e-10)


def test_input_checks_and_degenerate_data():
    X, _ = planted()
    with pytest.raises(ValueError, match="alpha"):
        TClust(alpha=1.0).fit(X)
    with pytest.raises(ValueError, match="restr_factor"):
        TClust(restr_factor=0.5).fit(X)
    with pytest.raises(ValueError, match="trims too many"):
        TClust(3, alpha=0.9).fit(X[:10])
    # Identical observations: every covariance is zero, so the identity is used.
    fit = TClust(2, alpha=0.0, n_init=5, max_iter=3, random_state=0).fit(np.ones((8, 2)))
    assert set(fit.labels_) <= {0, 1}
