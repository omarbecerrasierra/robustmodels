import itertools

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import reference as ref
from robustmodels.covariance import MCD, MRCD, MVE, OGK, MMEstimator, SEstimator, h_alpha_n
from robustmodels.scale import qn

ESTIMATORS = [
    MCD(random_state=0),
    MCD(algorithm="deterministic"),
    OGK(),
    SEstimator(n_trials=50, random_state=0),
    MMEstimator(n_trials=50, random_state=0),
    MVE(random_state=0),
    MRCD(),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def masked_problems():
    """Small samples with a tight outlier cluster shaped like the bulk."""
    rng = np.random.default_rng(7)
    for k in range(12):
        n, m = 14 + k % 3, int(round([0.2, 0.3, 0.4][k % 3] * (14 + k % 3)))
        A = rng.normal(size=(2, 2))
        bulk = rng.normal(size=(n - m, 2)) @ A
        out = rng.normal(size=2) * rng.uniform(2, 6) + 0.15 * rng.normal(size=(m, 2)) @ A
        yield np.vstack([bulk, out])


@pytest.mark.parametrize("X", list(masked_problems()))
def test_fast_mcd_finds_the_global_optimum(X):
    n, p = X.shape
    mcd = MCD(random_state=0).fit(X)
    found = np.linalg.slogdet(np.cov(X[mcd.raw_support_].T))[1]
    assert found == pytest.approx(ref.mcd_logdet(X, h_alpha_n(0.5, n, p)), abs=1e-9)


def test_univariate_mcd_is_exact():
    x = np.random.default_rng(1).standard_t(2, size=13)
    h = h_alpha_n(0.5, 13, 1)
    best = min(np.var(x[list(s)]) for s in itertools.combinations(range(13), h))
    mcd = MCD().fit(x[:, None])
    assert np.var(x[mcd.raw_support_]) == pytest.approx(best)


@pytest.mark.parametrize("est", [MCD(random_state=0), SEstimator(random_state=0),
                                 MMEstimator(random_state=0), MVE(random_state=0)])  # fmt: skip
def test_affine_equivariance(est):
    rng = np.random.default_rng(2)
    X = rng.standard_t(3, size=(60, 3))
    A, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    fit, fit_t = est.fit(X), type(est)(**est.get_params()).fit(X @ A + b)
    assert fit_t.location_ == pytest.approx(fit.location_ @ A + b)
    assert fit_t.covariance_ == pytest.approx(A.T @ fit.covariance_ @ A)


def test_mcd_small_sample_correction():
    diag = {True: [], False: []}
    for seed in range(150):
        X = np.random.default_rng(seed).normal(size=(40, 5))
        for corr in diag:
            fit = MCD(algorithm="deterministic", small_sample_correction=corr).fit(X)
            diag[corr].append(np.mean(np.diag(fit.covariance_)))
    assert np.mean(diag[True]) == pytest.approx(1, abs=0.05)
    assert np.mean(diag[False]) < 0.9


def corr(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


@pytest.mark.parametrize("est", ESTIMATORS, ids=lambda e: type(e).__name__)
def test_consistency_and_robustness(est):
    rng = np.random.default_rng(3)
    S = np.array([[2.0, 0.8, 0.0], [0.8, 1.0, 0.3], [0.0, 0.3, 0.5]])
    X = rng.multivariate_normal(np.zeros(3), S, size=2000)
    assert est.fit(X).covariance_ == pytest.approx(S, rel=0.2, abs=0.1)
    # 15% clustered outliers: location and shape must stay close (the scale of
    # S/MM is inflated, the known maximum bias of 50% breakdown S-estimators).
    X[:300] = rng.normal(loc=12, size=(300, 3))
    fit = est.fit(X)
    assert np.abs(fit.location_).max() < 0.3
    assert np.abs(corr(fit.covariance_) - corr(S)).max() < 0.1


def test_s_estimator_scale_equation():
    from robustmodels import scale

    X = np.random.default_rng(4).standard_t(4, size=(100, 2))
    s = SEstimator(random_state=0).fit(X)
    d = np.sqrt(s.dist_) * s.scale_  # distances w.r.t. the shape matrix
    c = scale._tuning_for_breakdown(0.5, 2)
    assert np.mean(scale._rho(d / s.scale_, c)) == pytest.approx(0.5, abs=1e-8)


def test_mrcd_high_dimensional():
    X = np.random.default_rng(5).normal(size=(20, 50))
    fit = MRCD().fit(X)
    scale = np.array([qn(col) for col in X.T])
    assert 0 < fit.rho_ < 1
    # The fixed regularization can leave the final subset below the bound.
    condition = np.linalg.cond(fit.covariance_ / np.outer(scale, scale))
    assert condition <= fit.max_condition * (1 + 1e-10)


def test_mrcd_is_mcd_when_well_conditioned():
    X = np.random.default_rng(6).normal(size=(100, 3))
    assert MRCD().fit(X).rho_ == 0


def test_exact_fit_warns():
    X = np.column_stack([np.arange(20.0), np.zeros(20)])
    X[:3, 1] = [5, -4, 7]
    with pytest.warns(RuntimeWarning, match="exact fit"):
        MCD(random_state=0).fit(X)


def test_invalid_parameters():
    X = np.random.default_rng(7).normal(size=(30, 2))
    for est in (MCD(alpha=0.3), MCD(algorithm="slow"), OGK(scale="iqr"), MRCD(rho=1.0)):
        with pytest.raises(ValueError):
            est.fit(X)


@pytest.mark.parametrize(
    "estimator, parameter",
    [
        (MCD(n_trials=0), "n_trials"),
        (MCD(n_trials=1.5), "n_trials"),
        (MVE(n_trials=-1), "n_trials"),
        (SEstimator(n_trials=0), "n_trials"),
        (SEstimator(n_best=0), "n_best"),
        (SEstimator(k_steps=0), "k_steps"),
        (SEstimator(max_iter=0), "max_iter"),
        (MMEstimator(max_iter=0), "max_iter"),
        (MMEstimator(efficiency=np.nan), "efficiency"),
        (OGK(n_iter=1.5), "n_iter"),
        (OGK(n_iter=True), "n_iter"),
        (OGK(beta=0), "beta"),
        (OGK(beta=1), "beta"),
        (OGK(beta=np.nan), "beta"),
        (MRCD(max_condition=1), "max_condition"),
        (MRCD(max_condition=np.inf), "max_condition"),
        (MRCD(alpha=np.nan), "alpha"),
        (MRCD(rho=np.nan), "rho"),
        (MVE(alpha=1), "alpha"),
    ],
)
def test_invalid_parameters_name_the_parameter(estimator, parameter):
    X = np.random.default_rng(17).normal(size=(20, 2))
    with pytest.raises(ValueError, match=parameter):
        estimator.fit(X)


def test_mcd_rejects_too_many_features():
    with pytest.raises(ValueError, match="use MRCD"):
        MCD().fit(np.eye(3))


def test_mrcd_singular_covariance_raises_a_clear_error():
    X = np.random.default_rng(21).normal(size=(10, 30))
    with pytest.raises(ValueError, match="larger rho"):
        MRCD(rho=0).fit(X)


def test_reweighting_that_keeps_every_point_gives_the_sample_covariance():
    # Checked against robustbase::covMcd, which applies no factor without truncation.
    X = np.random.default_rng(3).uniform(-1, 1, size=(12, 2))
    fitted = MCD(algorithm="deterministic").fit(X)
    assert fitted.support_.all()
    np.testing.assert_allclose(fitted.location_, X.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(fitted.covariance_, np.cov(X, rowvar=False))


def test_mrcd_rejects_a_single_observation_subset():
    with pytest.raises(ValueError, match="at least two"):
        MRCD().fit([[1.0], [2.0]])


def test_mrcd_constant_data_needs_explicit_regularization():
    X = np.ones((10, 3))
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="zero scatter"):
            MRCD().fit(X)
        fitted = MRCD(rho=0.1).fit(X)
    np.testing.assert_allclose(fitted.location_, np.ones(3))
    np.testing.assert_allclose(fitted.covariance_, 0.1 * np.eye(3))


def test_mrcd_accepts_a_constant_column_without_numeric_warnings():
    X = np.column_stack([np.arange(10), np.ones(10)])
    with np.errstate(all="raise"):
        fitted = MRCD().fit(X)
    assert np.isfinite(fitted.covariance_).all()
    assert np.linalg.eigvalsh(fitted.covariance_).min() > 0


@pytest.mark.filterwarnings("error")
def test_mrcd_full_coverage_of_two_observations():
    fitted = MRCD(alpha=1).fit([[0.0], [1.0]])
    np.testing.assert_allclose(fitted.location_, [0.5])
    np.testing.assert_allclose(fitted.covariance_, [[0.5]])


def test_mrcd_does_not_cap_required_regularization():
    from robustmodels.covariance import _min_rho

    covariance = np.diag([0.0, 1e8])
    rho = _min_rho(covariance, 50)
    regularized = rho * np.eye(2) + (1 - rho) * covariance
    assert rho > 1 - 1e-6
    assert np.linalg.cond(regularized) == pytest.approx(50, rel=1e-8)


@pytest.mark.parametrize(
    "estimator", [MVE(n_trials=2), SEstimator(n_trials=2), MMEstimator(n_trials=2)]
)
def test_singular_data_raises_a_clear_error(estimator):
    X = np.column_stack([np.arange(10.0), np.arange(10.0)])
    with pytest.raises(ValueError, match="hyperplane"):
        estimator.fit(X)


@pytest.mark.parametrize("scale", ["tau", "qn", "mad", "pn"])
def test_ogk_is_equivariant_under_column_rescaling(scale):
    X = np.random.default_rng(18).normal(size=(60, 3))
    factors = np.array([0.5, -2.0, 3.0])
    offset = np.array([3.0, 2.0, -1.0])
    original = OGK(scale=scale).fit(X)
    transformed = OGK(scale=scale).fit(X * factors + offset)
    np.testing.assert_allclose(
        transformed.location_, original.location_ * factors + offset, atol=1e-12
    )
    expected = original.covariance_ * np.outer(factors, factors)
    np.testing.assert_allclose(transformed.covariance_, expected, atol=1e-12)


@pytest.mark.parametrize(
    "estimator",
    [
        MCD(n_trials=10, random_state=0),
        MCD(algorithm="deterministic"),
        MVE(n_trials=10, random_state=0),
        OGK(),
        MRCD(),
        SEstimator(n_trials=10, random_state=0),
        MMEstimator(n_trials=10, random_state=0),
    ],
)
def test_refit_replaces_estimates_without_changing_input(estimator):
    rng = np.random.default_rng(19)
    first = rng.normal(size=(40, 2))
    second = rng.normal(size=(35, 3))
    before = second.copy()
    estimator.fit(first)
    assert estimator.fit(second) is estimator
    np.testing.assert_array_equal(second, before)
    assert estimator.n_features_in_ == 3
    assert estimator.location_.shape == (3,)
    assert estimator.covariance_.shape == (3, 3)
    np.testing.assert_allclose(estimator.dist_, estimator.mahalanobis(second), atol=1e-12)


def test_full_coverage_mcd_is_empirical_covariance():
    X = np.random.default_rng(20).normal(size=(30, 3))
    estimator = MCD(alpha=1, reweighting=False).fit(X)
    np.testing.assert_allclose(estimator.location_, X.mean(axis=0))
    np.testing.assert_allclose(estimator.covariance_, np.cov(X, rowvar=False))
    assert estimator.support_.all()


@pytest.mark.parametrize("eigenvalues", [[1.0, 2.0, 3.0], [0.0, 1.0, 100.0], [0.01, 1.0, 100.0]])
def test_mrcd_uses_the_smallest_regularization_for_the_condition_bound(eigenvalues):
    from robustmodels.covariance import _min_rho

    covariance = np.diag(eigenvalues)
    rho = _min_rho(covariance, 50)
    regularized = rho * np.eye(3) + (1 - rho) * covariance
    assert np.linalg.cond(regularized) <= 50 * (1 + 1e-12)
    if rho > 0:
        smaller = rho - 1e-6
        assert np.linalg.cond(smaller * np.eye(3) + (1 - smaller) * covariance) > 50
