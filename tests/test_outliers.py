import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.utils.estimator_checks import parametrize_with_checks

import reference as ref
from robustmodels.outliers import AdjustedOutlyingness, StahelDonohoOutlyingness, _whiskers
from robustmodels.scale import mad

DETECTORS = [StahelDonohoOutlyingness, AdjustedOutlyingness]


def expected_failures(estimator):
    # These checks expect some training points flagged, as with a fixed
    # contamination; the data-driven cutoff flags none in their clean data.
    reason = "the cutoff flags no point in the outlier-free check data"
    return {"check_outliers_train": reason, "check_outliers_fit_predict": reason}


@parametrize_with_checks(
    [
        StahelDonohoOutlyingness(random_state=0),
        AdjustedOutlyingness(n_directions=50, random_state=0),
    ],
    expected_failed_checks=expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def hazen_quantile(x, q):
    """Type 5 quantile: linear interpolation at position n q + 1/2 (1-based)."""
    x = np.sort(x)
    position = len(x) * q + 0.5
    below = int(np.floor(position))
    return x[below - 1] + (position - below) * (x[below] - x[below - 1])


def test_univariate_definitions():
    x = np.random.default_rng(0).lognormal(size=41)
    X = x[:, None]
    m = np.median(x)
    sdo = StahelDonohoOutlyingness().fit(X)
    np.testing.assert_allclose(sdo.outlyingness_, np.abs(x - m) / mad(x))
    y = x - m
    q1, q3 = hazen_quantile(y, 0.25), hazen_quantile(y, 0.75)
    mc = ref.medcouple(y)
    upper = y[y <= q3 + 1.5 * (q3 - q1) * np.exp(3 * mc)].max()
    lower = -y[y >= q1 - 1.5 * (q3 - q1) * np.exp(-4 * mc)].min()
    ao = AdjustedOutlyingness().fit(X)
    np.testing.assert_allclose(ao.outlyingness_, np.where(y > 0, y / upper, -y / lower))
    logs = np.log(0.1 + ao.outlyingness_)
    cutoff = np.exp(np.median(logs) + mad(logs) * stats.norm.ppf(0.995)) - 0.1
    assert ao.cutoff_ == pytest.approx(cutoff)


@pytest.mark.parametrize("detector", DETECTORS)
def test_affine_invariance(detector):
    rng = np.random.default_rng(1)
    X = rng.lognormal(size=(40, 3))
    A = np.array([[2.0, 1.0, 0.0], [0.5, -1.0, 3.0], [1.0, 0.0, -2.0]])
    base = detector(n_directions=100, random_state=0).fit(X)
    moved = detector(n_directions=100, random_state=0).fit(X @ A + [5.0, -3.0, 1.0])
    np.testing.assert_allclose(moved.outlyingness_, base.outlyingness_, rtol=1e-9)
    assert moved.cutoff_ == pytest.approx(base.cutoff_)


@pytest.mark.parametrize("detector", DETECTORS)
def test_outlyingness_is_convex(detector):
    rng = np.random.default_rng(2)
    fit = detector(n_directions=100, random_state=0).fit(rng.lognormal(size=(60, 2)))
    a, b = rng.normal(scale=3, size=(2, 200, 2))
    t = rng.uniform(size=(200, 1))
    between = fit.outlyingness(t * a + (1 - t) * b)
    chord = t[:, 0] * fit.outlyingness(a) + (1 - t[:, 0]) * fit.outlyingness(b)
    assert np.all(between <= chord + 1e-12)


def test_directions_approach_the_supremum_over_all_directions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    X[:3] += 4
    angles = np.linspace(0, np.pi, 100_000, endpoint=False)
    projections = X @ np.vstack([np.cos(angles), np.sin(angles)])
    distance = np.abs(projections - np.median(projections, axis=0))
    scale = np.median(distance, axis=0) / stats.norm.ppf(0.75)  # consistent MAD
    dense = (distance / scale).max(axis=1)
    fit = StahelDonohoOutlyingness(n_directions=1000).fit(X)  # all 435 pairs
    assert fit.directions_.shape[1] == 435
    ratio = fit.outlyingness_ / dense
    assert np.all((0.95 < ratio) & (ratio < 1 + 1e-3))


def test_adjusted_outlyingness_matches_exact_arithmetic():
    # stackloss is integer data: in 3D the normal through three points is an exact
    # integer cross product, so projections, ties and the outlyingness are exact.
    path = Path(__file__).parent / "data" / "r_reference.json"
    X = np.array(json.loads(path.read_text())["regression"]["stackloss"]["X"], dtype=np.int64)
    normals = [
        np.cross(X[j] - X[i], X[k] - X[i]) for i, j, k in itertools.combinations(range(21), 3)
    ]
    N = np.array([normal for normal in normals if normal.any()]).T
    y = (X @ N).astype(float)
    y -= np.median(y, axis=0)
    lower, upper = np.array([_whiskers(column) for column in y.T]).T
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(y == 0, 0, np.abs(y) / np.where(y > 0, upper, lower)).max(axis=1)
    fit = AdjustedOutlyingness(n_directions=10**6).fit(X.astype(float))  # all 1309 planes
    np.testing.assert_allclose(fit.outlyingness_, exact, rtol=1e-12)


def test_adjusted_outlyingness_finds_outliers_that_skewness_masks():
    X = np.random.default_rng(0).lognormal(size=(100, 2))
    X[:5] = [-3, -3]  # outliers on the short side
    ao = AdjustedOutlyingness(random_state=0).fit(X)
    np.testing.assert_array_equal(np.flatnonzero(ao.predict(X) == -1), np.arange(5))
    sdo = StahelDonohoOutlyingness(random_state=0).fit(X)
    assert not (sdo.predict(X)[:5] == -1).any()  # masked by the long tail


@pytest.mark.parametrize("detector", DETECTORS)
def test_detection_and_false_positives_at_the_normal_model(detector):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    clean = detector(n_directions=200, random_state=0).fit(X)
    assert np.mean(clean.predict(X) == -1) < 0.02
    X[:15] += [6.0, -6.0, 6.0]
    flagged = detector(n_directions=200, random_state=0).fit(X).predict(X) == -1
    assert flagged[:15].all() and flagged[15:].mean() < 0.02


@pytest.mark.parametrize("detector", DETECTORS)
def test_scores_new_observations_with_the_training_fit(detector):
    X = np.random.default_rng(5).normal(size=(80, 2))
    fit = detector(random_state=0).fit(X)
    new = np.array([[0.0, 0.0], [10.0, 10.0]])
    np.testing.assert_array_equal(fit.predict(new), [1, -1])
    depth = fit.score_samples(new)
    np.testing.assert_allclose(depth, 1 / (1 + fit.outlyingness(new)))
    np.testing.assert_allclose(fit.decision_function(new), depth - fit.offset_)
    assert fit.offset_ == pytest.approx(1 / (1 + fit.cutoff_))


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_majority_on_one_point_makes_the_rest_infinitely_outlying(detector):
    X = np.zeros((40, 2))
    X[30:] = np.random.default_rng(6).normal(size=(10, 2))
    fit = detector(random_state=0).fit(X)
    assert np.all(fit.outlyingness_[:30] == 0)
    assert np.all(np.isinf(fit.outlyingness_[30:]))
    np.testing.assert_array_equal(fit.predict(X), np.where(np.arange(40) < 30, 1, -1))


def test_random_directions_are_reproducible():
    X = np.random.default_rng(7).normal(size=(60, 3))
    first = StahelDonohoOutlyingness(n_directions=40, random_state=0).fit(X)
    second = StahelDonohoOutlyingness(n_directions=40, random_state=0).fit(X)
    assert first.directions_.shape == (3, 40)
    np.testing.assert_array_equal(first.directions_, second.directions_)
    np.testing.assert_allclose(np.linalg.norm(first.directions_, axis=0), 1)


@pytest.mark.parametrize(
    "detector, X, message",
    [
        (StahelDonohoOutlyingness(), np.ones((3, 3)) + np.eye(3), "more observations"),
        (StahelDonohoOutlyingness(n_directions=0), np.eye(4)[:, :2], "n_directions"),
        (AdjustedOutlyingness(), np.outer(np.arange(10.0), [1.0, 2.0, 3.0]), "singular"),
    ],
)
def test_invalid_input_raises_a_clear_error(detector, X, message):
    with pytest.raises(ValueError, match=message):
        detector.fit(X)
