"""The numbers behind each diagnostic plot; appearance is not tested."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from scipy.stats import chi2, norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from robustmodels import plots
from robustmodels.covariance import MCD
from robustmodels.multivariate import ROBPCA, RobustLDA
from robustmodels.outliers import StahelDonohoOutlyingness


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


PCA2 = SimpleNamespace(n_components_=2, components_=np.eye(3)[:2])


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


def test_pca_outlier_map_classifies_score_and_orthogonal_distances():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2)) @ rng.normal(size=(2, 6)) + 0.1 * rng.normal(size=(80, 6))
    X[:4] += 3  # away from the plane
    fit = ROBPCA(2, random_state=0).fit(X)
    points = groups(plots.pca_outlier_map(fit))
    far = fit.score_distances_ > fit.score_cutoff_
    off = fit.orthogonal_distances_ > fit.orthogonal_cutoff_
    assert off[:4].all()
    both = np.column_stack([fit.score_distances_, fit.orthogonal_distances_])
    classes = [
        ("Regular", ~far & ~off),
        ("Orthogonal outliers", ~far & off),
        ("Good leverage", far & ~off),
        ("Bad leverage", far & off),
    ]
    assert set(points) == {f"{label} ({mask.sum()})" for label, mask in classes if mask.any()}
    for label, mask in classes:
        if mask.any():
            np.testing.assert_allclose(points[f"{label} ({mask.sum()})"], both[mask])
    new = np.vstack([fit.location_, fit.location_ + 50])
    assert set(groups(plots.pca_outlier_map(fit, new))) == {"Regular (1)", "Bad leverage (1)"}


def test_scree_plot_and_biplot_show_variances_scores_and_loadings():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2)) @ rng.normal(size=(2, 5)) + 0.2 * rng.normal(size=(60, 5))
    X[:3] += 4
    fit = ROBPCA(3, random_state=0).fit(X)
    scree = plots.scree_plot(fit)
    np.testing.assert_allclose(scree.lines[0].get_ydata(), fit.explained_variance_)
    share = np.cumsum(fit.explained_variance_) / fit.explained_variance_.sum()
    assert [text.get_text() for text in scree.texts[:3]] == [f"{value:.0%}" for value in share]
    ax = plots.biplot(fit, X, components=(0, 2), feature_names=list("abcde"), n_arrows=2)
    points = groups(ax)
    assert sum(len(group) for group in points.values()) == 60
    flagged = ~fit.support_
    np.testing.assert_allclose(
        points[f"Flagged ({flagged.sum()})"], fit.transform(X)[flagged][:, [0, 2]]
    )
    loadings = fit.components_[[0, 2]].T
    longest = np.argsort(-np.linalg.norm(loadings, axis=1))[:2]
    names = [text.get_text() for text in ax.texts if text.get_text() in list("abcde")]
    assert names == [list("abcde")[j] for j in longest]
    full = ROBPCA(2, random_state=0).fit(X[:, :2])  # no orthogonal distances
    assert sum(len(group) for group in groups(plots.biplot(full, X[:, :2])).values()) == 60


def test_discriminant_plot_marks_classes_and_errors():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(40, 2)), rng.normal(size=(40, 2)) + 4])
    y = np.repeat(["a", "b"], 40)
    fit = RobustLDA(random_state=0).fit(X, y)
    clean = groups(plots.discriminant_plot(fit, X, y))
    assert set(clean) == {"a (40)", "b (40)"}
    np.testing.assert_allclose(clean["b (40)"], X[40:])
    ellipse = plots.discriminant_plot(fit, X, y).lines[0].get_xydata() - fit.means_[0]
    squared = np.sum(ellipse * np.linalg.solve(fit.covariance_, ellipse.T).T, axis=1)
    np.testing.assert_allclose(squared, chi2.ppf(0.975, 2))
    X[0] = [4, 4]  # the rule assigns it to the other class
    points = groups(plots.discriminant_plot(fit, X, y, feature_names=["u", "v"]))
    np.testing.assert_allclose(points["Misclassified (1)"], [[4, 4]])


def test_quantile_plots_pair_sorted_values_with_reference_quantiles():
    X = np.random.default_rng(0).normal(size=(50, 3))
    X[:4] += 6
    fit = MCD(random_state=0).fit(X)
    points = groups(plots.distance_qq(fit, X))
    distances = np.sqrt(fit.mahalanobis(X))
    flagged = distances > np.sqrt(chi2.ppf(0.975, 3))
    plotted = np.vstack(list(points.values()))
    plotted = plotted[np.argsort(plotted[:, 1])]
    expected = np.sqrt(chi2.ppf((np.arange(50) + 0.5) / 50, 3))
    np.testing.assert_allclose(plotted, np.column_stack([expected, np.sort(distances)]))
    assert len(points[f"Flagged ({flagged.sum()})"]) == flagged.sum() >= 4
    y = np.r_[np.linspace(-2, 2, 49), 9.0]
    model = SimpleNamespace(scale_=2, predict=lambda X: np.zeros(len(X)))
    points = groups(plots.residual_qq(model, X, 2 * y))
    assert set(points) == {"Regular (49)", "Flagged (1)"}
    np.testing.assert_allclose(points["Flagged (1)"], [[norm.ppf(49.5 / 50), 9]])
    model.predict = lambda X: X[:, 0]
    points = groups(plots.residual_plot(model, X, X[:, 0] + 2 * y))
    np.testing.assert_allclose(points["Flagged (1)"], [[X[-1, 0], 9]])
    np.testing.assert_allclose(points["Regular (49)"], np.column_stack([X[:49, 0], y[:49]]))


@pytest.mark.filterwarnings("ignore:exact fit")
def test_singular_covariance_is_not_given_a_chi_square_cutoff():
    x = np.r_[np.arange(10.0), 104.5]
    X = np.column_stack([x, x])  # rank 1: the pseudo-inverse ignores the other direction
    X[-1, 1] = -95.5
    fit = MCD(random_state=0).fit(X)
    for call in (
        lambda: plots.distance_distance(fit, X),
        lambda: plots.tolerance_ellipse(fit, X),
        lambda: plots.distance_qq(fit, X),
        lambda: plots.outlier_map(SimpleNamespace(scale_=1, predict=lambda X: X[:, 0]), X, x),
    ):
        with pytest.raises(ValueError, match="covariance is singular"):
            call()


def test_outlyingness_plot_marks_infinite_scores():
    fit = SimpleNamespace(outlyingness_=np.array([0.2, 2.0, np.inf, np.inf]), cutoff_=1.0)
    points = groups(plots.outlyingness_plot(fit))
    assert set(points) == {"Regular (1)", "Flagged (1)", "Infinite (2)"}
    assert np.all(np.isfinite(points["Infinite (2)"])) and np.all(points["Infinite (2)"][:, 1] > 2)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: plots.outlier_map(SimpleNamespace(scale_=0), [[1], [2]], [1, 2]), "exact fit"),
        (lambda: plots.outlier_map(SimpleNamespace(scale_=1), np.ones((20, 1)), [1]), "1 values"),
        (lambda: plots.residual_plot(None, None, None, id_n=0), "id_n"),
        (lambda: plots.residual_qq(SimpleNamespace(scale_=1), np.ones((20, 1)), [1]), "1 values"),
        (lambda: plots.biplot(SimpleNamespace(n_components_=1), None), "two different"),
        (lambda: plots.biplot(PCA2, np.ones((3, 3)), components=(1, 1)), "two different"),
        (lambda: plots.biplot(PCA2, np.ones((3, 3)), feature_names=["a"]), "3 labels"),
        (lambda: plots.biplot(None, None, n_arrows=0), "n_arrows"),
        (lambda: plots.discriminant_plot(None, np.ones((3, 3)), [1, 2, 3]), "two variables"),
        (lambda: plots.discriminant_plot(None, np.ones((3, 2)), [1, 2]), "2 values"),
        (
            lambda: plots.discriminant_plot(None, np.ones((3, 2)), [1, 2, 3], feature_names="a"),
            "labels",
        ),
        (
            lambda: plots.pca_outlier_map(ROBPCA(2).fit(np.arange(60.0).reshape(30, 2) ** 2 % 7)),
            "no orthogonal",
        ),
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


def test_bagplot_plots_its_statistics_and_counts():
    from robustmodels.depth import bagplot_stats

    X = np.random.default_rng(0).normal(size=(40, 2))
    X[:2] = [[6, 6], [-7, 6]]
    stats = bagplot_stats(X)
    ax = plots.bagplot(X, feature_names=["a", "b"])
    points = groups(ax)
    np.testing.assert_array_equal(points[f"Outliers ({stats.outliers.sum()})"], X[stats.outliers])
    np.testing.assert_array_equal(
        points[f"Regular ({(~stats.outliers).sum()})"], X[~stats.outliers]
    )
    np.testing.assert_allclose(ax.patches[0].get_xy()[:-1], stats.loop)
    np.testing.assert_allclose(ax.patches[1].get_xy()[:-1], stats.bag)
    assert ax.get_xlabel() == "a"
    with pytest.raises(ValueError, match="feature names"):
        plots.bagplot(X, feature_names=["one"])
    with pytest.raises(ValueError, match="id_n"):
        plots.bagplot(X, id_n=0)


def test_directional_plot_label():
    from robustmodels.outliers import DirectionalOutlyingness

    fit = DirectionalOutlyingness().fit(np.arange(20.0)[:, None])
    assert plots.outlyingness_plot(fit).get_ylabel() == "Directional outlyingness"
