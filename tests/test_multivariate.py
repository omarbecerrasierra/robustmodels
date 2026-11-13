"""Robust PCA, discriminant analysis and tests on mean vectors.

Reference values: tests/data/r_multivariate.json, from rrcov and pcaPP (its
"meta" entry records the versions). PcaHubert uses every pair of observations
as a direction when there are at most 250 pairs, so those fits are
deterministic and comparable. rrcov's orthogonal cutoff uses the median and
MAD instead of the univariate MCD of the paper and is not compared.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.estimator_checks import parametrize_with_checks

from robustmodels.covariance import MCD
from robustmodels.multivariate import (
    ROBPCA,
    ProjectionPursuitPCA,
    RobustLDA,
    RobustQDA,
    _n_components,
    _pair_outlyingness,
    _spatial_median,
    _t2_approximation,
    _unimcd,
    _wilks,
    hotelling_t2,
    manova,
)
from robustmodels.scale import mad, qn

R = json.loads((Path(__file__).parent / "data" / "r_multivariate.json").read_text())
PCA, GROUPS = R["pca"], R["groups"]


def close(a, b, tol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b).max() <= tol * np.abs(b).max()


def expected_failures(estimator):
    if isinstance(estimator, RobustQDA):
        # This check fits iris, where more than half of the setosa petal widths are tied.
        return {"check_positive_only_tag_during_fit": "exact fit: a class of iris is singular"}
    return {}


@parametrize_with_checks(
    [
        ROBPCA(random_state=0),
        ProjectionPursuitPCA(n_components=1),
        RobustLDA(random_state=0),
        RobustQDA(random_state=0),
    ],
    expected_failed_checks=expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def plane_with_outliers(n=100, p=20, outliers=10):
    """A noisy plane in ``p`` dimensions with outliers far from it."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2)) @ rng.normal(size=(2, p)) + 0.1 * rng.normal(size=(n, p))
    X[:outliers] += 5
    return X


@pytest.mark.parametrize("name", PCA)
@pytest.mark.parametrize("key, k", [("k0", None), ("k2", 2)])
def test_robpca_matches_rrcov(name, key, k):
    r = PCA[name][key]
    X = np.array(PCA[name]["X"])
    fit = ROBPCA(k, random_state=0).fit(X)
    assert fit.n_components_ == r["k"]
    loadings = np.reshape(r["loadings"], (X.shape[1], -1)).T
    signs = np.sign(np.sum(loadings * fit.components_, axis=1))  # rrcov does not fix the signs
    assert close(fit.components_, loadings * signs[:, None], 1e-9)
    assert close(fit.location_, r["center"], 1e-10)
    assert close(fit.explained_variance_, r["eigenvalues"], 1e-10)
    assert close(fit.score_distances_, r["sd"], 1e-9)
    assert np.abs(fit.orthogonal_distances_ - r["od"]).max() <= 1e-9 * np.abs(X).max()


def test_robpca_recovers_the_subspace_and_flags_the_outliers():
    X = plane_with_outliers()
    fit = ROBPCA(random_state=0).fit(X)
    assert fit.n_components_ == 2  # chosen by the 80% rule
    clean = X[10:] - X[10:].mean(axis=0)
    classical = np.linalg.svd(clean, full_matrices=False)[2][:2]
    assert np.linalg.svd(fit.components_ @ classical.T, compute_uv=False).min() > 0.999
    assert np.all(fit.orthogonal_distances_[:10] > fit.orthogonal_cutoff_)
    assert not fit.support_[:10].any() and fit.support_[10:].mean() > 0.9
    # The classical first component is pulled towards the outliers.
    pulled = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[2][:2]
    assert np.linalg.svd(pulled @ classical.T, compute_uv=False).min() < 0.9


def test_robpca_is_orthogonally_equivariant():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6)) * [4, 3, 2, 1, 1, 1]  # 190 pairs: every direction is used
    Q = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    base = ROBPCA(2, random_state=0).fit(X)
    moved = ROBPCA(2, random_state=0).fit(3 * X @ Q + 5)
    np.testing.assert_allclose(moved.location_, 3 * base.location_ @ Q + 5, atol=1e-9)
    np.testing.assert_allclose(moved.explained_variance_, 9 * base.explained_variance_)
    np.testing.assert_allclose(np.abs(moved.components_), np.abs(base.components_ @ Q), atol=1e-9)
    np.testing.assert_allclose(moved.score_distances_, base.score_distances_)
    np.testing.assert_allclose(moved.orthogonal_distances_, 3 * base.orthogonal_distances_)
    assert moved.orthogonal_cutoff_ == pytest.approx(3 * base.orthogonal_cutoff_)


def test_robpca_with_all_components_reconstructs_the_data():
    X = np.random.default_rng(2).normal(size=(12, 3))  # too few rows for the direct MCD
    fit = ROBPCA(3, random_state=0).fit(X)
    np.testing.assert_allclose(fit.inverse_transform(fit.transform(X)), X)
    np.testing.assert_allclose(fit.components_ @ fit.components_.T, np.eye(3), atol=1e-12)
    assert fit.orthogonal_cutoff_ == 0
    np.testing.assert_array_equal(fit.support_, fit.score_distances_ <= fit.score_cutoff_)


def test_robpca_distances_follow_their_definitions():
    X = plane_with_outliers()
    fit = ROBPCA(2, random_state=0).fit(X)
    new = np.random.default_rng(3).normal(size=(5, 20))
    scores = (new - fit.location_) @ fit.components_.T
    score, orthogonal = fit.distances(new)
    np.testing.assert_allclose(score, np.sqrt(np.sum(scores**2 / fit.explained_variance_, axis=1)))
    residual = new - fit.location_ - scores @ fit.components_
    np.testing.assert_allclose(orthogonal, np.linalg.norm(residual, axis=1))
    assert fit.score_cutoff_ == pytest.approx(np.sqrt(stats.chi2.ppf(0.975, 2)))
    center, scale = _unimcd(fit.orthogonal_distances_ ** (2 / 3), fit.h_)
    assert fit.orthogonal_cutoff_ == pytest.approx((center + scale * stats.norm.ppf(0.975)) ** 1.5)
    assert np.all(np.abs(fit.components_).max(axis=1) == fit.components_.max(axis=1))


def test_robpca_random_directions_are_reproducible():
    X = plane_with_outliers()  # 4950 pairs: 250 random ones are used
    first, second = (ROBPCA(2, random_state=0).fit(X) for _ in range(2))
    np.testing.assert_array_equal(first.components_, second.components_)


def test_number_of_components_rule():
    assert _n_components(np.array([5.0, 3.0, 1.5, 0.5]), 4) == 2  # 80% reached
    assert _n_components(np.array([5.0, 4.0, 3.0, 2.0]), 2) == 2  # capped
    assert _n_components(np.array([5.0, 1.0, 1e-4, 1e-5]), 4) == 1
    assert _n_components(np.array([1.0, 1e-4, 1.0]), 3) == 1  # stops at the first tiny one


def test_univariate_mcd_of_robpca():
    y = np.r_[np.random.default_rng(4).normal(size=40), 50.0, 60.0]
    center, scale = _unimcd(y, 32)
    assert abs(center) < 0.5 and 0.5 < scale < 1.5
    np.testing.assert_allclose(_unimcd(y, len(y)), (y.mean(), y.std(ddof=1)))
    assert _unimcd(np.r_[np.zeros(8), 1.0, 2.0], 8) == (0.0, 0.0)


def test_pair_outlyingness_skips_directions_without_spread():
    Z = np.zeros((10, 2))
    Z[8:] = [[1.0, 0.0], [0.0, 1.0]]  # eight tied points: every projection has zero scale
    np.testing.assert_array_equal(_pair_outlyingness(Z, 8, 250, None), np.zeros(10))


def test_projection_pursuit_with_the_standard_deviation_is_classical_pca():
    X = np.random.default_rng(5).normal(size=(60, 4)) @ np.diag([4, 3, 2, 1])
    fit = ProjectionPursuitPCA(4, scale=np.std).fit(X - X.mean(axis=0) + _spatial_median(X))
    values, vectors = np.linalg.eigh(np.cov(X.T, ddof=0))
    np.testing.assert_allclose(fit.explained_variance_, values[::-1], rtol=1e-6)
    np.testing.assert_allclose(np.abs(fit.components_ @ vectors[:, ::-1]), np.eye(4), atol=1e-3)


@pytest.mark.parametrize("scale_name, scale_fn", [("mad", mad), ("qn", qn)])
def test_projection_pursuit_maximizes_the_scale_in_the_plane(scale_name, scale_fn):
    X = np.random.default_rng(6).normal(size=(50, 2)) @ np.array([[3.0, 1.0], [0.0, 1.0]])
    X[:5] += [0, 15]
    fit = ProjectionPursuitPCA(2, scale=scale_name).fit(X)
    angles = np.linspace(0, np.pi, 20_000, endpoint=False)
    centered = X - _spatial_median(X)
    dense = [scale_fn(centered @ [np.cos(a), np.sin(a)]) ** 2 for a in angles[::20]]
    assert fit.explained_variance_[0] >= max(dense) * (1 - 1e-3)
    assert fit.components_[0] @ fit.components_[1] == pytest.approx(0, abs=1e-12)
    second = scale_fn(centered @ fit.components_[1]) ** 2
    assert fit.explained_variance_[1] == pytest.approx(second)


@pytest.mark.parametrize("name", PCA)
@pytest.mark.parametrize("scale", ["mad", "qn"])
def test_projection_pursuit_is_not_worse_than_pcapp(name, scale):
    X = np.array(PCA[name]["X"])
    assert close(_spatial_median(X), PCA[name]["l1median"], 1e-6)
    fit = ProjectionPursuitPCA(1, scale=scale).fit(X)
    # Both searches stop at local maxima of a criterion that is not smooth.
    assert fit.explained_variance_[0] >= PCA[name][f"grid_{scale}"] * 0.75


def test_projection_pursuit_resists_outliers_that_rotate_classical_pca():
    X = np.random.default_rng(0).normal(size=(100, 3)) * [5, 1, 1]
    X[:10] = [0, 100, 0]
    classical = np.linalg.svd(X - X.mean(axis=0))[2][0]
    assert abs(classical[1]) > 0.99
    fit = ProjectionPursuitPCA(1).fit(X)
    assert abs(fit.components_[0, 0]) > 0.99
    assert fit.transform(X).shape == (100, 1)
    assert fit.orthogonal_cutoff_ > 0 and fit.h_ == 75


def test_spatial_median_minimizes_the_sum_of_distances():
    X = np.random.default_rng(7).lognormal(size=(40, 3))
    median = _spatial_median(X)
    total = np.linalg.norm(X - median, axis=1).sum()
    for shift in 1e-4 * np.vstack([np.eye(3), -np.eye(3)]):
        assert total <= np.linalg.norm(X - median - shift, axis=1).sum()
    np.testing.assert_allclose(_spatial_median(np.ones((5, 2))), [1, 1])


@pytest.mark.parametrize("name", GROUPS)
def test_discriminant_rules_match_rrcov(name):
    r = GROUPS[name]
    X, y = np.array(r["X"]), np.array(r["y"])
    lda = RobustLDA(random_state=0).fit(X, y)
    qda = RobustQDA(random_state=0).fit(X, y)
    assert np.mean(lda.predict(X) == np.array(r["lda"]["pred"])) > 0.97
    assert np.mean(qda.predict(X) == np.array(r["qda"]["pred"])) > 0.97
    if name == "hemophilia":  # both FAST-MCD searches find the same subsets
        assert close(lda.means_, r["lda"]["means"], 1e-12)
        assert close(lda.covariance_, r["lda"]["cov"], 1e-12)
        assert close(qda.means_, r["qda"]["means"], 1e-12)
        assert close(qda.covariances_, r["qda"]["cov"], 1e-12)


def two_classes(outliers=0):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(60, 2)), rng.normal(size=(60, 2)) + 3])
    X[:outliers] = [-25, -25]  # mislabeled-looking outliers in the first class
    return X, np.repeat(["a", "b"], 60)


def test_lda_follows_the_pooled_mcd_definition():
    X, y = two_classes()
    fit = RobustLDA(random_state=0).fit(X, y)
    labels = (y == "b").astype(int)
    centers = np.array([MCD(random_state=0).fit(X[labels == k]).location_ for k in (0, 1)])
    pooled = MCD(random_state=0).fit(X - centers[labels])
    np.testing.assert_allclose(fit.means_, centers + pooled.location_)
    np.testing.assert_allclose(fit.covariance_, pooled.covariance_)
    np.testing.assert_array_equal(fit.support_, pooled.support_)
    np.testing.assert_allclose(fit.priors_, [0.5, 0.5])
    new = np.random.default_rng(1).normal(size=(6, 2)) + 1.5
    precision = np.linalg.inv(fit.covariance_)
    linear = new @ precision @ fit.means_.T - 0.5 * np.sum(fit.means_ @ precision * fit.means_, 1)
    np.testing.assert_allclose(fit.decision_function(new), linear[:, 1] - linear[:, 0])
    np.testing.assert_array_equal(fit.predict(new), np.where(linear[:, 1] > linear[:, 0], "b", "a"))


def test_qda_follows_the_normal_log_density():
    X, y = two_classes()
    X[y == "b"] *= 2
    fit = RobustQDA(priors=[0.3, 0.7], random_state=0).fit(X, y)
    new = np.random.default_rng(1).normal(size=(6, 2)) * 3
    density = [
        stats.multivariate_normal(mean, covariance).pdf(new) * prior
        for mean, covariance, prior in zip(fit.means_, fit.covariances_, fit.priors_, strict=True)
    ]
    posterior = np.array(density).T / np.sum(density, axis=0)[:, None]
    np.testing.assert_allclose(fit.predict_proba(new), posterior)
    np.testing.assert_allclose(fit.decision_function(new), np.log(density[1] / density[0]))


@pytest.mark.parametrize("rule", [RobustLDA, RobustQDA])
def test_discriminant_rules_resist_outliers(rule):
    X, y = two_classes(outliers=12)
    clean, labels = two_classes()
    robust = rule(random_state=0).fit(X, y)
    assert np.abs(robust.means_ - [[0, 0], [3, 3]]).max() < 0.5
    assert np.mean(robust.predict(clean) == labels) > 0.95
    classical = LinearDiscriminantAnalysis().fit(X, y)
    assert np.abs(classical.means_[0]).max() > 4  # dragged towards the outliers


def test_discriminant_rules_with_three_classes():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(size=(40, 2)) + shift for shift in ([0, 0], [4, 0], [0, 4])])
    y = np.repeat([10, 20, 30], 40)
    for rule in (RobustLDA, RobustQDA):
        fit = rule(random_state=0).fit(X, y)
        assert fit.decision_function(X).shape == (120, 3)
        np.testing.assert_allclose(fit.predict_proba(X).sum(axis=1), 1)
        assert np.mean(fit.predict(X) == y) > 0.9
    never = RobustLDA(priors=[0.5, 0.5, 0.0], random_state=0).fit(X, y)
    assert not (never.predict(X) == 30).any()


@pytest.mark.parametrize("name", R["t2"])
def test_hotelling_matches_rrcov(name):
    r = R["t2"][name]
    result = hotelling_t2(np.array(PCA[name]["X"]), r["mu"], random_state=0)
    assert result.statistic == pytest.approx(r["T2"], rel=1e-10)
    assert result.f_statistic == pytest.approx(r["F"], rel=1e-10)
    assert result.df == pytest.approx(r["df"], rel=1e-10)
    assert result.pvalue == pytest.approx(r["p"], rel=1e-9)
    assert close(result.location, r["center"], 1e-12)


def test_hotelling_approximation_constants():
    for row in R["t2_dq"]:
        d, q = _t2_approximation(int(row[0]), int(row[1]))
        assert (d, q) == pytest.approx(row[2:], rel=1e-12)
    # For large n the statistic tends to kappa * chi2_p, with kappa -> 1.26 for p = 1.
    d, q = _t2_approximation(1, 10**7)
    assert d == pytest.approx(1.2583, abs=1e-3) and q > 1e5


def test_hotelling_holds_its_level_and_resists_outliers():
    rng = np.random.default_rng(8)
    pvalues = [hotelling_t2(rng.normal(size=(40, 1))).pvalue for _ in range(400)]
    assert 0.02 < np.mean(np.array(pvalues) < 0.05) < 0.09
    X = rng.normal(size=(60, 3))
    X[:6] += 8
    assert hotelling_t2(X, random_state=0).pvalue > 0.05
    assert hotelling_t2(X, [1, 1, 1], random_state=0).pvalue < 1e-3
    assert stats.ttest_1samp(X[:, 0], 0).pvalue < 0.05  # the classical test is misled


@pytest.mark.parametrize("name", GROUPS)
def test_wilks_lambda_matches_rrcov(name):
    r = GROUPS[name]
    X, labels = np.array(r["X"]), np.unique(r["y"], return_inverse=True)[1]
    statistic, means = _wilks(X, labels, MCD(random_state=0).fit)  # FAST-MCD, as rrcov
    assert statistic == pytest.approx(r["wilks"], rel=1e-10)
    assert close(means, r["wilks_means"], 1e-10)


def test_manova_follows_its_definition():
    rng = np.random.default_rng(9)
    X, groups = rng.normal(size=(60, 2)), np.repeat(["a", "b", "c"], 20)
    X[:3] += 10
    result = manova(X, groups, n_simulations=40, random_state=0)
    labels = np.unique(groups, return_inverse=True)[1]
    fit_mcd = MCD(algorithm="deterministic").fit
    centers = np.array([fit_mcd(X[labels == k]).location_ for k in range(3)])
    kept = fit_mcd(X - centers[labels]).dist_ < stats.chi2.ppf(0.975, 2)
    assert not kept[:3].any()
    Z, z = X[kept], labels[kept]
    means = np.array([Z[z == k].mean(axis=0) for k in range(3)])
    within = sum((Z[z == k] - means[k]).T @ (Z[z == k] - means[k]) for k in range(3))
    total = (Z - Z.mean(axis=0)).T @ (Z - Z.mean(axis=0))
    assert result.statistic == pytest.approx(np.linalg.det(within) / np.linalg.det(total))
    np.testing.assert_allclose(result.means, means)
    simulation = np.random.default_rng(0)
    logs = np.log([_wilks(simulation.normal(size=X.shape), labels, fit_mcd)[0] for _ in range(40)])
    q = 2 * logs.mean() ** 2 / logs.var(ddof=1)
    assert result.df == pytest.approx(q)
    assert result.chi2_statistic == pytest.approx(np.log(result.statistic) * q / logs.mean())
    assert result.pvalue == pytest.approx(stats.chi2.sf(result.chi2_statistic, q))


def test_manova_detects_a_shift_and_ignores_outliers():
    rng = np.random.default_rng(0)
    X, groups = rng.normal(size=(90, 2)), np.repeat([0, 1, 2], 30)
    X[:4] += 9
    assert manova(X, groups, n_simulations=100, random_state=0).pvalue > 0.05
    X[groups == 2] += 1.5
    assert manova(X, groups, n_simulations=100, random_state=0).pvalue < 1e-3


def test_manova_needs_observations_in_every_group():
    X = np.random.default_rng(1).normal(size=(46, 2))
    X[:6] *= 1000  # a small group so spread out that all of it is outlying
    groups = np.r_[np.zeros(6), np.ones(40)]
    with pytest.raises(ValueError, match="no observations left"):
        manova(X, groups, n_simulations=2)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda X, y: ROBPCA(0).fit(X), "n_components"),
        (lambda X, y: ROBPCA(alpha=0.4).fit(X), "alpha"),
        (lambda X, y: ROBPCA(n_directions=0).fit(X), "n_directions"),
        (lambda X, y: ROBPCA().fit(np.ones((10, 3))), "identical"),
        (lambda X, y: ProjectionPursuitPCA(3).fit(X), "n_components"),
        (lambda X, y: ProjectionPursuitPCA(scale="sd").fit(X), "scale"),
        (lambda X, y: ProjectionPursuitPCA(alpha=2).fit(X), "alpha"),
        (lambda X, y: ProjectionPursuitPCA(2).fit(X * [1, 0]), "zero robust scale"),
        (lambda X, y: RobustLDA(alpha=0.2).fit(X, y), "alpha"),
        (lambda X, y: RobustQDA(alpha=0.2).fit(X, y), "alpha"),
        (lambda X, y: RobustLDA().fit(X, np.zeros(len(X))), "two classes"),
        (lambda X, y: RobustLDA().fit(X[:62], y[:62]), "more observations than variables"),
        (lambda X, y: RobustLDA(priors=[0.5]).fit(X, y), "one nonnegative number"),
        (lambda X, y: RobustQDA(priors=[0.5, 0.6]).fit(X, y), "sum to 1"),
        (lambda X, y: hotelling_t2(X, [0, 0, 0]), "mu must contain 2"),
        (lambda X, y: manova(X, y, n_simulations=1), "n_simulations"),
        (lambda X, y: manova(X, y, alpha=0.3), "alpha"),
    ],
)
def test_invalid_input_raises_a_clear_error(call, message):
    X, y = two_classes()
    with pytest.raises(ValueError, match=message):
        call(X, y)


@pytest.mark.filterwarnings("ignore:exact fit")
def test_singular_covariances_raise_a_clear_error():
    X, y = two_classes()
    flat = np.column_stack([X[:, 0], X[:, 0]])
    with pytest.raises(ValueError, match="pooled covariance is singular"):
        RobustLDA(random_state=0).fit(flat, y)
    with pytest.raises(ValueError, match="covariance of a class is singular"):
        RobustQDA(random_state=0).fit(flat, y)
    with pytest.raises(ValueError, match="MCD covariance is singular"):
        hotelling_t2(flat, random_state=0)
