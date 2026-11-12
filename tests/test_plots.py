"""The numbers behind each diagnostic plot; appearance is not tested."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from robustmodels import plots
from robustmodels.covariance import MCD
from robustmodels.outliers import StahelDonohoOutlyingness


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def groups(ax):
    """Plotted points by legend label."""
    return {
        artist.get_label(): np.asarray(artist.get_offsets())
        for artist in ax.collections
        if isinstance(artist, PathCollection)
    }


def test_outlier_map_classifies_residuals_and_leverage():
    X = np.r_[np.linspace(-1, 1, 40), 10, 12]
    residuals = np.zeros(42)
    residuals[[0, -1]] = [3, -4]  # one vertical outlier and one bad leverage point
    model = SimpleNamespace(scale_=2, predict=lambda X: np.zeros(len(X)))
    points = groups(plots.outlier_map(model, X, 2 * residuals, random_state=0))
    assert set(points) == {
        "Regular (39)",
        "Vertical outliers (1)",
        "Good leverage (1)",
        "Bad leverage (1)",
    }
    distances = np.sqrt(MCD(random_state=0).fit(X[:, None]).dist_)
    np.testing.assert_allclose(points["Bad leverage (1)"], [[distances[-1], -4]])
    np.testing.assert_allclose(points["Good leverage (1)"], [[distances[-2], 0]])
    np.testing.assert_allclose(points["Vertical outliers (1)"], [[distances[0], 3]])


def test_distance_distance_plots_classical_and_robust_distances():
    X = np.random.default_rng(0).normal(size=(60, 2))
    X[:5] += 8
    fit = MCD(random_state=0).fit(X)
    points = groups(plots.distance_distance(fit, X))
    centered = X - X.mean(axis=0)
    classical = np.sqrt(np.sum(centered * np.linalg.solve(np.cov(X.T), centered.T).T, axis=1))
    robust = np.sqrt(fit.mahalanobis(X))
    flagged = robust > np.sqrt(chi2.ppf(0.975, 2))
    for label, mask in (("Regular", ~flagged), ("Flagged", flagged)):
        expected = np.column_stack([classical[mask], robust[mask]])
        np.testing.assert_allclose(points[f"{label} ({mask.sum()})"], expected)


def test_tolerance_ellipses_follow_the_two_covariances():
    X = np.random.default_rng(0).normal(size=(80, 2))
    X[:8] += [5, -4]
    fit = MCD(random_state=0).fit(X)
    lines = {line.get_label(): line for line in plots.tolerance_ellipse(fit, X).lines}
    for label, center, covariance in [
        ("Robust 97.5% ellipse", fit.location_, fit.covariance_),
        ("Classical 97.5% ellipse", X.mean(axis=0), np.cov(X.T)),
    ]:
        points = lines[label].get_xydata() - center
        squared = np.sum(points * np.linalg.solve(covariance, points.T).T, axis=1)
        np.testing.assert_allclose(squared, chi2.ppf(0.975, 2))


def test_adjusted_boxplot_follows_its_definition():
    np.testing.assert_array_equal(plots._fivenum(np.arange(8.0)), [0, 1.5, 3.5, 5.5, 7])
    x = np.random.default_rng(1).lognormal(size=100)
    s, reflected = plots._adjusted_box_stats(x), plots._adjusted_box_stats(-x)
    inside = x[(x >= s["fences"][0]) & (x <= s["fences"][1])]
    assert (s["whislo"], s["whishi"]) == (inside.min(), inside.max())
    assert len(inside) + len(s["fliers"]) == len(x)
    # A reflected sample has the opposite medcouple and mirrored fences.
    assert reflected["mc"] == pytest.approx(-s["mc"])
    np.testing.assert_allclose(reflected["fences"], -np.array(s["fences"])[::-1])
    _, ax = plt.subplots()
    assert plots.adjusted_boxplot(np.column_stack([x, -x]), labels=["x", "-x"], ax=ax) is ax
    assert plots.adjusted_boxplot(x).get_xticklabels()[0].get_text() == "1"


def test_outlyingness_plot_scores_new_observations():
    fit = StahelDonohoOutlyingness(random_state=0).fit(
        np.random.default_rng(0).normal(size=(50, 2))
    )
    new = np.array([[0.0, 0.0], [20.0, 20.0]])
    points = groups(plots.outlyingness_plot(fit, new))
    assert set(points) == {"Regular (1)", "Flagged (1)"}
    np.testing.assert_allclose(points["Flagged (1)"], [[1, fit.outlyingness(new)[1]]])


def test_outlyingness_plot_marks_infinite_scores():
    fit = SimpleNamespace(outlyingness_=np.array([0.2, 2.0, np.inf, np.inf]), cutoff_=1.0)
    points = groups(plots.outlyingness_plot(fit))
    assert set(points) == {"Regular (1)", "Flagged (1)", "Infinite (2)"}
    assert np.all(np.isfinite(points["Infinite (2)"])) and np.all(points["Infinite (2)"][:, 1] > 2)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: plots.outlier_map(SimpleNamespace(scale_=0), [[1], [2]], [1, 2]), "exact fit"),
        (lambda: plots.tolerance_ellipse(None, [[1], [2]]), "two variables"),
        (lambda: plots.tolerance_ellipse(None, [[1, 2], [3, 4]], feature_names=["a"]), "labels"),
        (lambda: plots.adjusted_boxplot([]), "nonempty"),
        (lambda: plots.adjusted_boxplot([1.0, np.nan]), "NaN"),
        (lambda: plots.adjusted_boxplot([[1, 2], [3, 4]], labels=["a"]), "zip"),
        (lambda: plots.distance_distance(None, None, id_n=0), "id_n"),
    ],
)
def test_invalid_input_raises_a_clear_error(call, message):
    with pytest.raises(ValueError, match=message):
        call()
