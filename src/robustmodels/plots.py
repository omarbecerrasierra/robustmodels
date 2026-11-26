"""Diagnostic plots for robust estimators, drawn with matplotlib.

Every function takes a fitted estimator (or data), draws on ``ax`` or on a new
figure, and returns the matplotlib ``Axes``. Save a figure for publication with
``ax.figure.savefig("figure.pdf")``. The style is set on the returned axes only:
global matplotlib settings are not changed.

Colors keep one role across the plots: robust estimates in blue, flagged
outliers in orange, classical estimates and good leverage points in aqua,
regular observations in gray. Each class also has its own marker, so the plots
remain readable in grayscale.
"""

import numpy as np
from scipy.stats import chi2, norm, rankdata
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import check_array

from ._validation import as_1d, check_count, check_range
from .covariance import MCD, _logdet
from .depth import bagplot_stats, halfspace_depth
from .outliers import _adjusted_fences
from .scale import medcouple

__all__ = [
    "adjusted_boxplot",
    "bagplot",
    "biplot",
    "cluster_plot",
    "coefficient_plot",
    "ctl_curves",
    "dd_plot",
    "discriminant_plot",
    "distance_distance",
    "distance_qq",
    "hdr_plot",
    "log_concave_plot",
    "outlier_map",
    "outlyingness_plot",
    "pca_outlier_map",
    "residual_plot",
    "residual_qq",
    "score_plot",
    "scree_plot",
    "tolerance_ellipse",
]

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e8e7e1"
AXIS = "#b9b8ae"
REGULAR = "#9b9992"
BLUE = "#2a78d6"  # robust estimates
ORANGE = "#eb6834"  # flagged outliers
AQUA = "#1baf7a"  # classical estimates, good leverage points
VIOLET = "#7a5fd0"
_CLASSES = [(BLUE, "o"), (AQUA, "s"), (VIOLET, "^"), (SECONDARY, "v")]  # orange marks errors

_FIGSIZE = (5.2, 3.8)


def outlier_map(model, X, y, *, id_n=3, random_state=None, ax=None):
    """Regression outlier map (Rousseeuw & van Zomeren 1990).

    Standardized residuals ``(y - model.predict(X)) / model.scale_`` against
    robust distances of ``X`` from :class:`~robustmodels.covariance.MCD`.
    Residuals beyond +-2.5 and distances beyond the square root of the 0.975
    chi-square quantile split the observations into regular observations,
    vertical outliers, good leverage points and bad leverage points.

    Parameters
    ----------
    model : fitted regressor with ``predict`` and ``scale_``
        For example :class:`~robustmodels.regression.MMRegressor`.
    X : array-like of shape (n_samples, n_features)
        Predictors of full rank; ``ValueError`` if their MCD covariance is
        singular, because the chi-square cutoff does not apply.
    y : array-like of shape (n_samples,)
        One response per row of ``X``.
    id_n : int, default 3
        Positive maximum number of eligible observations labeled with their
        zero-based row positions. Severity combines residual and predictor distance.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the MCD fit for the distances.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created by default.

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Rousseeuw, P. J. and van Zomeren, B. C. (1990). Unmasking multivariate
    outliers and leverage points. *Journal of the American Statistical
    Association*, 85(411), 633-639.
    """
    check_count("id_n", id_n)
    X, residuals = _standardized_residuals(model, X, y)
    mcd = MCD(random_state=random_state).fit(X)
    _check_full_rank(mcd)
    distances = np.sqrt(mcd.dist_)
    d_cut, r_cut = np.sqrt(chi2.ppf(0.975, X.shape[1])), 2.5
    far, large = distances > d_cut, np.abs(residuals) > r_cut
    ax = _axes(ax)
    x_max = 1.08 * max(distances.max(), d_cut * 1.25)
    y_min = min(residuals.min(), -r_cut * 1.4)
    y_max = max(residuals.max(), r_cut * 1.4)
    y_pad = 0.08 * (y_max - y_min)
    y_min, y_max = y_min - y_pad, y_max + y_pad
    ax.set(xlim=(0, x_max), ylim=(y_min, y_max))
    for low, high in ((y_min, -r_cut), (r_cut, y_max)):  # shaded outlier zones
        ax.fill_between([0, d_cut], low, high, color=BLUE, alpha=0.06, lw=0, zorder=0)
        ax.fill_between([d_cut, x_max], low, high, color=ORANGE, alpha=0.08, lw=0, zorder=0)
    ax.fill_between([d_cut, x_max], -r_cut, r_cut, color=AQUA, alpha=0.06, lw=0, zorder=0)
    groups = [
        ("Regular", ~far & ~large, REGULAR, "o"),
        ("Vertical outliers", ~far & large, BLUE, "s"),
        ("Good leverage", far & ~large, AQUA, "^"),
        ("Bad leverage", far & large, ORANGE, "D"),
    ]
    for label, mask, color, marker in groups:
        _points(ax, distances[mask], residuals[mask], color, marker, f"{label} ({mask.sum()})")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for value in (-r_cut, r_cut):
        _cutoff(ax, value, "h", rf"$\pm${r_cut:g}" if value > 0 else None)
    _cutoff(ax, d_cut, "v", rf"$\sqrt{{\chi^2_{{{X.shape[1]},0.975}}}} = {d_cut:.2f}$")
    severity = np.maximum(np.abs(residuals) / r_cut, distances / d_cut)
    _label_extremes(ax, distances, residuals, severity, id_n, severity > 1)
    ax.set_xlabel("Robust distance of $x_i$ (MCD)")
    ax.set_ylabel(r"Standardized residual  $r_i / \hat\sigma$")
    _title(ax, "Regression outlier map", type(model).__name__)
    _legend(ax)
    return ax


def distance_distance(model, X, *, id_n=3, ax=None):
    """Distance-distance plot (Rousseeuw & Van Driessen 1999).

    Robust distances from a fitted covariance estimator against classical
    Mahalanobis distances. Large robust distances paired with small classical
    distances suggest masking. Both axes show unsquared distances; the classical
    covariance uses divisor ``n - 1``. Lines mark the square root of the 0.975
    chi-square quantile.

    Parameters
    ----------
    model : fitted covariance estimator with ``mahalanobis``
        For example :class:`~robustmodels.covariance.MCD`. ``ValueError`` if its
        covariance is singular, because the chi-square cutoff does not apply.
    X : array-like of shape (n_samples, n_features)
    id_n : int, default 3
        Positive maximum number of flagged robust distances labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Rousseeuw, P. J. and Van Driessen, K. (1999). A fast algorithm for the
    minimum covariance determinant estimator. *Technometrics*, 41(3), 212-223.
    """
    check_count("id_n", id_n)
    X = np.asarray(X, dtype=float)
    _check_full_rank(model)
    robust = np.sqrt(model.mahalanobis(X))
    classical = np.sqrt(_classical_distances(X))
    cut = np.sqrt(chi2.ppf(0.975, X.shape[1]))
    flagged = robust > cut
    ax = _axes(ax)
    top = 1.15 * max(robust.max(), classical.max(), cut * 1.25)  # room for labels
    ax.set(xlim=(0, top), ylim=(0, top))
    ax.plot([0, top], [0, top], color=AXIS, lw=0.9, zorder=1)
    _points(
        ax, classical[~flagged], robust[~flagged], REGULAR, "o", f"Regular ({(~flagged).sum()})"
    )
    _points(ax, classical[flagged], robust[flagged], ORANGE, "D", f"Flagged ({flagged.sum()})")
    label = rf"$\sqrt{{\chi^2_{{{X.shape[1]},0.975}}}} = {cut:.2f}$"
    _cutoff(ax, cut, "h", label)
    _cutoff(ax, cut, "v", None)
    _label_extremes(ax, classical, robust, robust, id_n, flagged)
    ax.set_xlabel("Classical Mahalanobis distance")
    ax.set_ylabel(f"Robust distance ({type(model).__name__})")
    _title(ax, "Distance-distance plot", "large robust distances can reveal masked outliers")
    _legend(ax)
    return ax


def _classical_distances(X):
    """Squared Mahalanobis distances to the mean, divisor ``n - 1``, from the correlations.

    Standardizing first keeps columns of very different scales (where a
    pseudo-inverse of the covariance drops the small ones) accurate.
    """
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    correlation = np.atleast_2d(np.corrcoef(Z, rowvar=False))
    return np.einsum("ij,ij->i", Z, np.linalg.solve(correlation, Z.T).T)


def tolerance_ellipse(model, X, *, feature_names=None, id_n=3, ax=None):
    """Robust and classical 97.5% tolerance ellipses for two variables.

    The robust ellipse uses the location and covariance of a fitted covariance
    estimator; the classical one, the sample mean and covariance. Observations
    with a robust distance beyond the ellipse are flagged. The classical
    covariance has divisor ``n - 1``. These are normal-reference contours,
    not confidence regions for the center or exact finite-sample tolerance regions.

    Parameters
    ----------
    model : fitted covariance estimator with ``location_`` and ``covariance_``
        ``ValueError`` if its covariance is singular: the ellipse would
        degenerate and the 97.5% label would not hold.
    X : array-like of shape (n_samples, 2)
    feature_names : sequence of two str, optional
        Exactly two axis labels; ``x_1`` and ``x_2`` by default.
    id_n : int, default 3
        Positive maximum number of flagged robust distances labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    check_count("id_n", id_n)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"tolerance_ellipse needs two variables, got shape {X.shape}")
    names = ["$x_1$", "$x_2$"] if feature_names is None else list(feature_names)
    if len(names) != 2:
        raise ValueError("feature_names must contain two labels")
    _check_full_rank(model)
    radius = np.sqrt(chi2.ppf(0.975, 2))
    robust = np.sqrt(model.mahalanobis(X))
    flagged = robust > radius
    classical = EmpiricalCovariance().fit(X)
    ax = _axes(ax)
    for center, cov, color, style, label in (
        (model.location_, model.covariance_, BLUE, "-", "Robust 97.5% ellipse"),
        (
            classical.location_,
            classical.covariance_ * len(X) / (len(X) - 1),
            AQUA,
            (0, (5, 2.5)),
            "Classical 97.5% ellipse",
        ),
    ):
        x, y = _ellipse(center, cov, radius)
        ax.plot(x, y, color=color, lw=1.8, ls=style, label=label, zorder=2, solid_capstyle="round")
        ax.plot(*center, marker="+", ms=9, mew=1.6, color=color, zorder=4)
    _points(ax, X[~flagged, 0], X[~flagged, 1], REGULAR, "o", f"Regular ({(~flagged).sum()})")
    _points(ax, X[flagged, 0], X[flagged, 1], ORANGE, "D", f"Flagged ({flagged.sum()})")
    _label_extremes(ax, X[:, 0], X[:, 1], robust, id_n, flagged)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    _title(ax, "Tolerance ellipses", f"{type(model).__name__} against the sample covariance")
    _legend(ax, below=True)
    return ax


def adjusted_boxplot(x, *, labels=None, ax=None):
    """Skewness-adjusted boxplot (Hubert & Vandervieren 2008).

    The whiskers reach the most extreme observations inside the fences
    ``Q1 - 1.5 exp(-4 MC) IQR`` and ``Q3 + 1.5 exp(3 MC) IQR`` for medcouple
    ``MC >= 0`` (``exp(-3 MC)`` and ``exp(4 MC)`` otherwise); observations
    beyond them are drawn individually. The box spans Tukey's hinges, as R's
    ``robustbase::adjbox``. Each box is labeled with its medcouple (MC).

    Parameters
    ----------
    x : array-like of shape (n_samples,) or (n_samples, n_variables)
        Finite, nonempty data; one box per column.
    labels : sequence of str, optional
        Box labels.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Hubert, M. and Vandervieren, E. (2008). An adjusted boxplot for skewed
    distributions. *Computational Statistics & Data Analysis*, 52(12),
    5186-5201.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim not in (1, 2) or data.size == 0:
        raise ValueError("x must be a nonempty 1-D or 2-D array")
    columns = [as_1d(data)] if data.ndim == 1 else [as_1d(c) for c in data.T]
    stats = [_adjusted_box_stats(column) for column in columns]
    names = [str(i + 1) for i in range(len(stats))] if labels is None else list(labels)
    ax = _axes(ax)
    ax.bxp(
        [
            {**s, "label": f"{name}\nMC = {s['mc']:.2f}"}
            for s, name in zip(stats, names, strict=True)
        ],
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": BLUE + "1f", "edgecolor": BLUE, "linewidth": 1.3},
        medianprops={"color": INK, "linewidth": 1.8, "solid_capstyle": "butt"},
        whiskerprops={"color": SECONDARY, "linewidth": 1.1},
        capprops={"color": SECONDARY, "linewidth": 1.1},
        flierprops={
            "marker": "D",
            "markersize": 4.5,
            "markerfacecolor": ORANGE,
            "markeredgecolor": "white",
            "markeredgewidth": 0.6,
        },
    )
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.08)
    n_out = sum(len(s["fliers"]) for s in stats)
    _title(ax, "Adjusted boxplot", f"{n_out} observations beyond the skewness-adjusted fences")
    return ax


def outlyingness_plot(detector, X=None, *, id_n=3, ax=None):
    """Outlyingness of each observation with the detector's cutoff.

    Parameters
    ----------
    detector : fitted :class:`~robustmodels.outliers.StahelDonohoOutlyingness`,
        :class:`~robustmodels.outliers.AdjustedOutlyingness` or
        :class:`~robustmodels.outliers.DirectionalOutlyingness`
    X : array-like of shape (n_samples, n_features), optional
        Observations to score against the fitted detector; training data by
        default. The cutoff is retained from training, without refitting.
    id_n : int, default 3
        Positive maximum number of flagged scores labeled with their zero-based
        row positions. Tied scores keep input order.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Infinite scores are drawn as triangles above the finite scores and cutoff;
    their display height is not a finite estimate. Legend counts separate finite
    flagged scores from infinite scores; the subtitle includes both. Empty
    classes are omitted.
    """
    check_count("id_n", id_n)
    values = detector.outlyingness_ if X is None else detector.outlyingness(X)
    index = np.arange(len(values))
    finite = np.isfinite(values)
    flagged = values > detector.cutoff_
    top = 1.15 * max(values[finite].max(initial=0), detector.cutoff_, np.finfo(float).eps)
    shown = np.where(finite, values, top)
    ax = _axes(ax)
    ax.vlines(index, 0, shown, color=GRID, lw=0.8, zorder=1)
    _points(ax, index[~flagged], shown[~flagged], REGULAR, "o", f"Regular ({(~flagged).sum()})")
    shown_flags = flagged & finite
    _points(
        ax, index[shown_flags], shown[shown_flags], ORANGE, "D", f"Flagged ({shown_flags.sum()})"
    )
    _points(ax, index[~finite], shown[~finite], ORANGE, "^", f"Infinite ({(~finite).sum()})")
    _cutoff(ax, detector.cutoff_, "h", f"cutoff = {detector.cutoff_:.2f}")
    ax.set_ylim(0, top * 1.08)
    ax.set_xlim(-0.02 * len(values) - 1, 1.02 * len(values))
    _label_extremes(ax, index, shown, values, id_n, flagged)
    ax.set_xlabel("Observation index")
    kinds = {"Adjusted": "Adjusted", "Directional": "Directional"}
    measure = f"{kinds.get(type(detector).__name__[:-12], 'Stahel-Donoho')} outlyingness"
    ax.set_ylabel(measure)
    _title(ax, measure, f"{flagged.sum()} of {len(values)} observations above the cutoff")
    _legend(ax)
    return ax


def pca_outlier_map(model, X=None, *, id_n=3, ax=None):
    """PCA outlier map (Hubert, Rousseeuw & Vanden Branden 2005).

    Orthogonal distances to the principal subspace against score distances
    inside it. The two cutoffs of the fitted model split the observations into
    regular observations, good leverage points (far inside the subspace),
    orthogonal outliers (far from the subspace) and bad leverage points (both).

    Parameters
    ----------
    model : fitted :class:`~robustmodels.multivariate.ROBPCA` or
        :class:`~robustmodels.multivariate.ProjectionPursuitPCA`
        It must keep fewer components than the rank of the data; otherwise
        there are no orthogonal distances and ``ValueError`` is raised.
    X : array-like of shape (n_samples, n_features), optional
        Observations to place on the map; training data by default. The
        cutoffs are retained from training.
    id_n : int, default 3
        Positive maximum number of flagged observations labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Hubert, M., Rousseeuw, P. J. and Vanden Branden, K. (2005). ROBPCA: a new
    approach to robust principal component analysis. *Technometrics*, 47(1),
    64-79.
    """
    check_count("id_n", id_n)
    s_cut, o_cut = model.score_cutoff_, model.orthogonal_cutoff_
    if not o_cut > 0:
        raise ValueError("the components span the data: there are no orthogonal distances")
    score, orthogonal = model.score_distances_, model.orthogonal_distances_
    if X is not None:
        score, orthogonal = model.distances(X)
    far, off = score > s_cut, orthogonal > o_cut
    ax = _axes(ax)
    x_max = 1.08 * max(score.max(), s_cut * 1.25)
    y_max = 1.3 * max(orthogonal.max(), o_cut * 1.4)  # room for the cutoff label
    ax.set(xlim=(0, x_max), ylim=(0, y_max))
    ax.fill_between([0, s_cut], o_cut, y_max, color=BLUE, alpha=0.06, lw=0, zorder=0)
    ax.fill_between([s_cut, x_max], o_cut, y_max, color=ORANGE, alpha=0.08, lw=0, zorder=0)
    ax.fill_between([s_cut, x_max], 0, o_cut, color=AQUA, alpha=0.06, lw=0, zorder=0)
    groups = [
        ("Regular", ~far & ~off, REGULAR, "o"),
        ("Orthogonal outliers", ~far & off, BLUE, "s"),
        ("Good leverage", far & ~off, AQUA, "^"),
        ("Bad leverage", far & off, ORANGE, "D"),
    ]
    for label, mask, color, marker in groups:
        _points(ax, score[mask], orthogonal[mask], color, marker, f"{label} ({mask.sum()})")
    _cutoff(ax, o_cut, "h", f"cutoff = {o_cut:.2f}")
    k = model.n_components_
    _cutoff(ax, s_cut, "v", rf"$\sqrt{{\chi^2_{{{k},0.975}}}} = {s_cut:.2f}$")
    severity = np.maximum(score / s_cut, orthogonal / o_cut)
    _label_extremes(ax, score, orthogonal, severity, id_n, severity > 1)
    ax.set_xlabel(f"Score distance ({k} components)")
    ax.set_ylabel("Orthogonal distance")
    _title(ax, "PCA outlier map", type(model).__name__)
    _legend(ax)
    return ax


def scree_plot(model, *, ax=None):
    """Scree plot: the robust variance of each principal component.

    Each point is labeled with the cumulative share of the variance of the
    fitted components, so fit as many components as you want to compare.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.multivariate.ROBPCA` or
        :class:`~robustmodels.multivariate.ProjectionPursuitPCA`
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    variances = np.asarray(model.explained_variance_)
    number = np.arange(1, len(variances) + 1)
    share = np.cumsum(variances) / variances.sum()
    ax = _axes(ax)
    ax.plot(number, variances, color=BLUE, lw=1.6, zorder=2)
    _points(ax, number, variances, BLUE, "o", None)
    for k, variance, cumulative in zip(number, variances, share, strict=True):
        ax.annotate(
            f"{cumulative:.0%}", (k, variance), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8, color=SECONDARY, zorder=4,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"},
        )  # fmt: skip
    ax.set_xticks(number)
    ax.set(xlim=(0.5, len(variances) + 0.5), ylim=(0, 1.18 * variances.max()))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Robust variance")
    _title(ax, "Scree plot", "labels: cumulative share of the variance of these components")
    return ax


def bagplot(
    X, *, factor=3.0, feature_names=None, id_n=3, max_samples=200, random_state=None, ax=None
):
    """Bagplot, the bivariate boxplot of Rousseeuw, Ruts & Tukey (1999).

    The cross is the Tukey median, the dark bag holds the half of the
    observations with the largest halfspace depth and the light loop is the
    convex hull of the observations inside the fence, the bag inflated by
    ``factor`` from the median. Observations outside the fence are flagged. The
    shape of the bag and the loop shows location, spread, correlation and
    skewness, as the box and whiskers do for one variable.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    factor : float, default 3.0
        Inflation of the bag that gives the fence.
    feature_names : sequence of two str, optional
        Axis labels; ``x_1`` and ``x_2`` by default.
    id_n : int, default 3
        Maximum number of outliers labeled with their zero-based row positions,
        the farthest from the median after scaling by the sample ranges first.
    max_samples : int, default 200
        Subset size for the median and the bag of larger samples; see
        :func:`~robustmodels.depth.bagplot_stats`.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the subset.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    check_count("id_n", id_n)
    stats = bagplot_stats(X, factor=factor, max_samples=max_samples, random_state=random_state)
    X = np.asarray(X, dtype=float)
    names = ["$x_1$", "$x_2$"] if feature_names is None else [str(name) for name in feature_names]
    if len(names) != 2:
        raise ValueError(f"expected 2 feature names, got {len(names)}")
    flagged = stats.outliers
    ax = _axes(ax)
    ax.fill(*stats.loop.T, color=BLUE, alpha=0.14, lw=0, zorder=1, label="Loop")
    ax.fill(*stats.bag.T, color=BLUE, alpha=0.38, lw=0, zorder=1.2, label="Bag (50%)")
    fence = np.vstack([stats.fence, stats.fence[:1]])
    ax.plot(*fence.T, color=SECONDARY, lw=0.9, ls=(0, (4, 3)), zorder=1.5, label="Fence")
    _points(ax, *X[~flagged].T, REGULAR, "o", f"Regular ({(~flagged).sum()})")
    _points(ax, *X[flagged].T, ORANGE, "D", f"Outliers ({flagged.sum()})")
    ax.plot(*stats.center, marker="+", color=INK, ms=11, mew=1.6, ls="none", zorder=4,
            label="Tukey median")  # fmt: skip
    offsets = X - stats.center
    reach = np.linalg.norm(offsets / np.ptp(X, axis=0), axis=1)
    _label_extremes(ax, X[:, 0], X[:, 1], reach, id_n, flagged)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    _title(ax, "Bagplot", f"{flagged.sum()} of {len(X)} observations outside the fence")
    _legend(ax, below=True)
    return ax


def biplot(model, X, *, components=(0, 1), feature_names=None, n_arrows=10, id_n=3, ax=None):
    """Biplot of a robust PCA: scores of the observations and loadings of the variables.

    Points are the scores on two components. Arrows are the loadings of the
    variables on the same components, all multiplied by one factor so that
    the longest arrow reaches 80% of the largest score. Observations beyond the
    score or orthogonal cutoff of the model are flagged.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.multivariate.ROBPCA` or
        :class:`~robustmodels.multivariate.ProjectionPursuitPCA`
        With at least two components.
    X : array-like of shape (n_samples, n_features)
        Observations to draw, usually the training data.
    components : pair of int, default (0, 1)
        Zero-based components on the horizontal and vertical axes.
    feature_names : sequence of str, optional
        One name per variable; ``x_1``, ``x_2``, ... by default.
    n_arrows : int, default 10
        Positive maximum number of variables drawn, those with the longest arrows.
    id_n : int, default 3
        Positive maximum number of flagged observations labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Gabriel, K. R. (1971). The biplot graphic display of matrices with
    application to principal component analysis. *Biometrika*, 58(3), 453-467.
    """
    check_count("id_n", id_n)
    check_count("n_arrows", n_arrows)
    a, b = components
    if model.n_components_ < 2 or not 0 <= min(a, b) < max(a, b) < model.n_components_:
        raise ValueError("components must be two different components of the model")
    p = model.components_.shape[1]
    names = [f"$x_{{{j + 1}}}$" for j in range(p)] if feature_names is None else feature_names
    if len(names) != p:
        raise ValueError(f"feature_names must contain {p} labels")
    scores = model.transform(X)[:, [a, b]]
    score, orthogonal = model.distances(X)
    severity = score / model.score_cutoff_
    if model.orthogonal_cutoff_ > 0:
        severity = np.maximum(severity, orthogonal / model.orthogonal_cutoff_)
    flagged = severity > 1
    ax = _axes(ax)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    _points(ax, *scores[~flagged].T, REGULAR, "o", f"Regular ({(~flagged).sum()})")
    _points(ax, *scores[flagged].T, ORANGE, "D", f"Flagged ({flagged.sum()})")
    loadings = model.components_[[a, b]].T
    lengths = np.linalg.norm(loadings, axis=1)
    arrows = loadings * 0.8 * np.abs(scores).max() / lengths.max()
    for j in np.argsort(-lengths, kind="stable")[:n_arrows]:
        ax.annotate(
            "", arrows[j], (0, 0), zorder=4,
            arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.2, "shrinkA": 0, "shrinkB": 0},
        )  # fmt: skip
        ax.annotate(
            names[j], arrows[j], xytext=3 * np.sign(arrows[j]), textcoords="offset points",
            ha="left" if arrows[j, 0] >= 0 else "right", fontsize=8, color=BLUE, zorder=4,
        )  # fmt: skip
    _label_extremes(ax, scores[:, 0], scores[:, 1], severity, id_n, flagged)
    variances = model.explained_variance_
    ax.set_xlabel(f"PC{a + 1}  (robust variance {variances[a]:.3g})")
    ax.set_ylabel(f"PC{b + 1}  (robust variance {variances[b]:.3g})")
    _title(ax, "Biplot", f"{type(model).__name__}: scores and scaled loadings")
    _legend(ax)
    return ax


def discriminant_plot(model, X, y, *, feature_names=None, ax=None):
    """Classification regions of a discriminant rule for two variables.

    Shaded regions show the class the rule predicts at each point and gray
    lines the boundaries between classes. Crosses and colored ellipses are the
    robust center and 97.5% tolerance ellipse of each class, from the fitted
    covariance (one for all classes in linear discriminant analysis).
    Observations are drawn by their true class; those the rule assigns to
    another class are circled. The error in the subtitle is the share of the
    plotted observations that are misclassified: for the training data it is
    the apparent error, which is optimistic.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.multivariate.RobustLDA` or
        :class:`~robustmodels.multivariate.RobustQDA`
    X : array-like of shape (n_samples, 2)
    y : array-like of shape (n_samples,)
        True class of each observation.
    feature_names : sequence of two str, optional
        Axis labels; ``x_1`` and ``x_2`` by default.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"discriminant_plot needs two variables, got shape {X.shape}")
    if len(y) != len(X):
        raise ValueError(f"y has {len(y)} values for {len(X)} rows of X")
    names = ["$x_1$", "$x_2$"] if feature_names is None else list(feature_names)
    if len(names) != 2:
        raise ValueError("feature_names must contain two labels")
    classes = list(model.classes_)
    styles = [_CLASSES[k % len(_CLASSES)] for k in range(len(classes))]
    low, high = X.min(axis=0), X.max(axis=0)
    pad = 0.08 * (high - low)
    grid = np.meshgrid(*(np.linspace(low[j] - pad[j], high[j] + pad[j], 500) for j in (0, 1)))
    region = np.searchsorted(classes, model.predict(np.column_stack([g.ravel() for g in grid])))
    region = region.reshape(grid[0].shape)
    ax = _axes(ax)
    levels = np.arange(len(classes) + 1) - 0.5
    ax.contourf(*grid, region, levels=levels, colors=[c for c, _ in styles], alpha=0.1, zorder=0)
    ax.contour(*grid, region, levels=levels[1:-1], colors=SECONDARY, linewidths=0.9, zorder=1)
    wrong = model.predict(X) != y
    covariances = np.broadcast_to(model._covariances(), (len(classes), 2, 2))
    radius = np.sqrt(chi2.ppf(0.975, 2))
    for k, (label, (color, marker)) in enumerate(zip(classes, styles, strict=True)):
        mask = y == label
        _points(ax, X[mask, 0], X[mask, 1], color, marker, f"{label} ({mask.sum()})")
        ax.plot(*_ellipse(model.means_[k], covariances[k], radius), color=color, lw=1.3, zorder=2)
        ax.plot(*model.means_[k], marker="+", ms=10, mew=1.8, color=INK, zorder=5)
    if wrong.any():
        ax.scatter(
            X[wrong, 0], X[wrong, 1], s=90, facecolors="none", edgecolors=ORANGE,
            linewidths=1.3, label=f"Misclassified ({wrong.sum()})", zorder=4,
        )  # fmt: skip
    ax.set(xlim=grid[0][0, [0, -1]], ylim=grid[1][[0, -1], 0])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    error = f"{wrong.mean():.1%} of these observations misclassified"
    _title(ax, "Classification regions", f"{type(model).__name__}: {error}")
    _legend(ax, below=True)
    return ax


def cluster_plot(model, X, *, features=(0, 1), feature_names=None, ax=None):
    """Clusters and trimmed observations of a robust clustering, as R's ``plot.tclust``.

    Observations are drawn by cluster, the trimmed ones as orange diamonds.
    Crosses and ellipses are the center and 97.5% tolerance ellipse of each
    cluster's normal model in the two plotted variables. With more variables,
    the ellipses are the marginal ones of those two.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.cluster.TClust`
    X : array-like of shape (n_samples, n_features)
        The training data, or other data to assign with ``model.predict``.
    features : pair of int, default (0, 1)
        Columns of ``X`` on the axes.
    feature_names : sequence of two str, optional
        Axis labels; ``x_1`` and ``x_2`` (the column numbers plus one) by default.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Fritz, H., García-Escudero, L. A. and Mayo-Iscar, A. (2012). tclust: An R
    package for a trimming approach to cluster analysis. *Journal of
    Statistical Software*, 47(12), 1-26.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"cluster_plot needs at least two variables, got shape {X.shape}")
    features = list(features)
    if (
        len(features) != 2
        or len(set(features)) != 2
        or not all(0 <= j < X.shape[1] for j in features)
    ):
        raise ValueError(f"features must be two different columns of X, got {features}")
    names = [f"$x_{j + 1}$" for j in features] if feature_names is None else list(feature_names)
    if len(names) != 2:
        raise ValueError("feature_names must contain two labels")
    labels = model.predict(X)
    ax = _axes(ax)
    radius = np.sqrt(chi2.ppf(0.975, 2))
    block = np.ix_(features, features)
    for k in range(len(model.means_)):
        color, marker = _CLASSES[k % len(_CLASSES)]
        mask = labels == k
        _points(
            ax,
            X[mask, features[0]],
            X[mask, features[1]],
            color,
            marker,
            f"Cluster {k + 1} ({mask.sum()})",
        )
        center = model.means_[k, features]
        ax.plot(
            *_ellipse(center, model.covariances_[k][block], radius), color=color, lw=1.3, zorder=2
        )
        ax.plot(*center, marker="+", ms=10, mew=1.8, color=INK, zorder=5)
    trimmed = labels == -1
    _points(
        ax,
        X[trimmed, features[0]],
        X[trimmed, features[1]],
        ORANGE,
        "D",
        f"Trimmed ({trimmed.sum()})",
    )
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    subtitle = (
        f"{type(model).__name__}: {len(model.means_)} clusters, alpha = {model.alpha:g}, "
        f"eigenvalue ratio <= {model.restr_factor:g}"
    )
    _title(ax, "Robust clustering", subtitle)
    _legend(ax)
    return ax


def hdr_plot(model, *, probs=(0.25, 0.5, 0.75), feature_names=None, ax=None):
    """Highest density regions of a kernel density estimate (Hyndman 1996), as R's ``plot.kde``.

    The region of probability ``prob`` is ``{x : f(x) >= level}``, with the
    level from :meth:`~robustmodels.density.KernelDensity.hdr_level`. For two
    variables the regions are filled, darker for the smaller ones, inside
    their boundary contours, over the training observations; for one
    variable they are shaded under the density curve, over a rug of the
    observations. Separate regions of the same probability reveal modes.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.density.KernelDensity` with one or two variables
    probs : sequence of float in (0, 1), default (0.25, 0.5, 0.75)
    feature_names : sequence of str, optional
        Axis labels; ``x_1`` and ``x_2`` by default.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Hyndman, R. J. (1996). Computing and graphing highest density regions.
    *The American Statistician*, 50(2), 120-126.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    X, d = model.X_, model.n_features_in_
    if d > 2:
        raise ValueError(f"hdr_plot needs a model of one or two variables, got {d}")
    names = [f"$x_{j + 1}$" for j in range(d)] if feature_names is None else list(feature_names)
    if len(names) != d:
        raise ValueError(f"expected {d} feature names, got {len(names)}")
    probs = np.sort(np.atleast_1d(np.asarray(probs, dtype=float)))[::-1]  # largest region first
    levels = model.hdr_level(probs)
    alphas = np.linspace(0.16, 0.6, len(probs))
    pad = 3 * np.sqrt(np.diag(model.bandwidth_))
    low, high = X.min(axis=0) - pad, X.max(axis=0) + pad
    ax = _axes(ax)
    regions = [
        Patch(color=BLUE, alpha=alpha, lw=0, label=f"{prob:.0%} region")
        for prob, alpha in zip(probs, alphas, strict=True)
    ]
    if d == 1:
        grid = np.linspace(low[0], high[0], 512)
        density = np.exp(model.score_samples(grid[:, None]))
        for level, alpha in zip(levels, alphas, strict=True):
            ax.fill_between(grid, 0, density, where=density >= level, color=BLUE, alpha=alpha, lw=0)
        ax.plot(grid, density, color=BLUE, lw=1.6, zorder=3, label="Density estimate")
        ax.plot(X[:, 0], np.zeros(len(X)), "|", color=SECONDARY, ms=7, mew=0.7,
                transform=ax.get_xaxis_transform(), zorder=2)  # fmt: skip
        ax.set_ylim(0, 1.08 * density.max())
        ax.set_ylabel("Density")
    else:
        axes = [np.linspace(low[j], high[j], 160) for j in range(2)]
        grid = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 2)
        density = np.exp(model.score_samples(grid)).reshape(160, 160)
        bounds = np.r_[levels, max(density.max(), levels[-1]) * (1 + 1e-9)]
        colors = [to_rgba(BLUE, alpha) for alpha in alphas]
        ax.contourf(*axes, density, levels=bounds, colors=colors, zorder=1)
        contours = ax.contour(*axes, density, levels=levels, colors=BLUE, linewidths=0.9, zorder=2)
        labels = {level: f"{prob:.0%}" for level, prob in zip(levels, probs, strict=True)}
        ax.clabel(contours, fmt=labels, fontsize=7.5, colors=INK)
        ax.scatter(*X.T, s=7, color=REGULAR, lw=0, zorder=3, label=f"Observations ({len(X)})")
        ax.set_ylabel(names[1])
    ax.set_xlabel(names[0])
    handles = ax.get_legend_handles_labels()[0]
    ax.legend(handles=[*handles, *regions], frameon=False, fontsize=8, labelcolor=SECONDARY)
    _title(ax, "Highest density regions", f"normal kernel, {len(X)} observations")
    return ax


def log_concave_plot(model, *, log=False, feature_name=None, ax=None):
    """Log-concave maximum likelihood density over the histogram of the sample, as R's ``plot.dlc``.

    The estimate is drawn with its knots, the observations where the slope of
    the log density changes. With ``log=True`` the log density, which is
    piecewise linear and concave, is drawn instead.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.shape.LogConcaveDensity`
    log : bool, default False
    feature_name : str, optional
        Label of the horizontal axis; ``x`` by default.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Dümbgen, L. and Rufibach, K. (2011). logcondens: Computations related to
    univariate log-concave density estimation. *Journal of Statistical
    Software*, 39(6), 1-28.
    """
    x, knots = model.support_, model.knots_
    grid = np.union1d(np.linspace(x[0], x[-1], 512), knots)
    values = model.score_samples(grid)
    at_knots = model.score_samples(knots)
    ax = _axes(ax)
    if not log:
        edges = np.histogram_bin_edges(x, bins="auto")
        ax.hist(x, bins=edges, weights=model.weights_, density=True, color=GRID, edgecolor="white",
                lw=0.6, zorder=1, label="Histogram")  # fmt: skip
        values, at_knots = np.exp(values), np.exp(at_knots)
    ax.plot(grid, values, color=BLUE, lw=1.8, zorder=3, label="Log-concave estimate")
    _points(ax, knots, at_knots, BLUE, "o", f"Knots ({len(knots)})")
    ax.set_xlabel("x" if feature_name is None else str(feature_name))
    ax.set_ylabel("Log density" if log else "Density")
    _title(ax, "Log-concave density", f"maximum likelihood, {len(x)} distinct observations")
    _legend(ax)
    return ax


def score_plot(model, X, y, *, ax=None):
    """Antitonic score of a regression by score matching, as R's ``plot.asm``.

    The estimated score ``psi`` (minus the derivative of the convex loss)
    against the residual, over a rug of the residuals, with the score of a
    normal distribution with the MAD of the residuals for comparison: the
    least squares fit uses that straight line, while a bounded score limits
    the influence of large residuals. The horizontal axis covers the central
    98% of the residuals, widened by a quarter of that range on each side.

    Parameters
    ----------
    model : fitted :class:`~robustmodels.shape.ScoreMatchingRegressor`
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Feng, O. Y., Kao, Y.-C., Xu, M. and Samworth, R. J. (2024). Optimal
    convex M-estimation via score matching. arXiv:2403.16688.
    """
    X, residuals = _standardized_residuals(model, X, y)
    residuals = residuals * model.scale_
    low, high = np.quantile(residuals, [0.01, 0.99])
    grid = np.linspace(low - (high - low) / 4, high + (high - low) / 4, 600)
    psi = model.psi(grid)
    ax = _axes(ax)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.plot(grid, -grid / model.scale_**2, color=AQUA, lw=1.2, ls=(0, (5, 3)), zorder=2,
            label="Normal score (MAD scale)")  # fmt: skip
    ax.plot(grid, psi, color=BLUE, lw=1.8, zorder=3, label="Antitonic score")
    ax.plot(residuals, np.zeros(len(residuals)), "|", color=SECONDARY, ms=7, mew=0.7,
            transform=ax.get_xaxis_transform(), zorder=2)  # fmt: skip
    reach = 1.6 * np.abs(psi).max()
    ax.set(xlim=(grid[0], grid[-1]), ylim=(-reach, reach))
    ax.set_xlabel("Residual")
    ax.set_ylabel(r"Score $\psi$")
    _title(ax, "Antitonic score", f"estimated information {model.information_:.3g}")
    _legend(ax)
    return ax


def coefficient_plot(
    models, *, labels=None, X=None, y=None, feature_names=None, intercept=False, level=0.95, ax=None
):
    """Coefficient estimates of several fitted models with their confidence intervals.

    Each row is a coefficient and each model a marker with its interval
    ``estimate +- z * standard error``, ``z`` the normal quantile of
    ``level``. Passing ``X`` and ``y`` adds least squares, with its classical
    standard errors, as the reference. Differences between intervals are a
    diagnostic for sensitivity to the choice of estimator.

    Parameters
    ----------
    models : fitted model or sequence of fitted models
        Each with ``coef_``, ``intercept_`` and ``std_errors_``, such as
        :class:`~robustmodels.regression.MMRegressor`,
        :class:`~robustmodels.rank.RankRegressor`,
        :class:`~robustmodels.shape.ScoreMatchingRegressor` or the robust GLMs,
        all with the same predictors.
    labels : sequence of str, optional
        Model names; the class names by default.
    X, y : array-like, optional
        Data for the least squares reference.
    feature_names : sequence of str, optional
        Coefficient names; ``x0, x1, ...`` by default.
    intercept : bool, default False
        Also show the intercepts.
    level : float in (0, 1), default 0.95
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    if not models:
        raise ValueError("provide at least one fitted model")
    if (X is None) != (y is None):
        raise ValueError("provide both X and y for the least squares reference")
    check_range("level", level, 0, 1, low_open=True, high_open=True)
    names = [type(model).__name__ for model in models] if labels is None else list(labels)
    if len(names) != len(models):
        raise ValueError(f"expected {len(models)} labels, got {len(names)}")
    p = len(models[0].coef_)
    if any(len(model.coef_) != p for model in models):
        raise ValueError("the models have different numbers of coefficients")
    rows = [f"x{j}" for j in range(p)] if feature_names is None else list(feature_names)
    if len(rows) != p:
        raise ValueError(f"expected {p} feature names, got {len(rows)}")
    fits = []
    for model in models:
        errors = np.asarray(model.std_errors_, dtype=float)
        first = (model.intercept_, errors[0]) if len(errors) == p + 1 else (np.nan, np.nan)
        fits.append((np.r_[first[0], model.coef_], np.r_[first[1], errors[-p:]]))
    styles = [(BLUE, "o"), (VIOLET, "^"), (SECONDARY, "v"), (MUTED, "D")]
    styles = [styles[k % len(styles)] for k in range(len(models))]
    if X is not None:
        reference = _least_squares(X, y)
        if len(reference[0]) != p + 1:
            raise ValueError("the least squares reference must use the same number of predictors")
        fits.append(reference)
        names.append("Least squares")
        styles.append((AQUA, "s"))
    keep = slice(0 if intercept else 1, None)
    rows = ["(Intercept)", *rows][keep]
    z = norm.ppf((1 + level) / 2)
    ax = _axes(ax)
    position = np.arange(len(rows))[::-1].astype(float)
    width = 0.7 / len(fits)
    for k, ((estimates, errors), name, (color, marker)) in enumerate(
        zip(fits, names, styles, strict=True)
    ):
        offset = (len(fits) - 1) / 2 * width - k * width
        ax.errorbar(
            estimates[keep], position + offset, xerr=z * errors[keep], fmt=marker, color=color,
            ms=5.5, mec="white", mew=0.7, elinewidth=1.4, capsize=0, label=name, zorder=3,
        )  # fmt: skip
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set_yticks(position, rows)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Estimate")
    subtitle = f"{level:.0%} intervals: estimate ± {z:.2f} standard errors"
    _title(ax, "Coefficient estimates", subtitle)
    _legend(ax, below=True)
    return ax


def dd_plot(X1, X2, *, labels=("Sample 1", "Sample 2"), random_state=None, ax=None):
    """Depth-depth plot of two samples (Liu, Parelius & Singh 1999).

    Each observation of both samples is placed at its halfspace depth with
    respect to the first sample and to the second. When both samples come
    from one distribution the points lie near the diagonal. A difference in
    location bends them into a lens away from the diagonal, a difference in
    spread puts them on one side of it, and skewness or shape differences
    give other departures. :func:`~robustmodels.density.kde_test` gives the
    corresponding test.

    Parameters
    ----------
    X1, X2 : array-like of shape (n1, n_features) and (n2, n_features)
        One-dimensional arrays are samples of one variable.
    labels : pair of str, default ("Sample 1", "Sample 2")
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the directions of the approximate depth for three or more variables.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Liu, R. Y., Parelius, J. M. and Singh, K. (1999). Multivariate analysis by
    data depth: descriptive statistics, graphics and inference. *The Annals of
    Statistics*, 27(3), 783-858.
    """
    X1, X2 = (np.asarray(X, dtype=float) for X in (X1, X2))
    X1, X2 = (X[:, None] if X.ndim == 1 else X for X in (X1, X2))
    if X1.shape[1] != X2.shape[1]:
        raise ValueError(f"the samples have {X1.shape[1]} and {X2.shape[1]} variables")
    labels = [str(label) for label in labels]
    if len(labels) != 2:
        raise ValueError(f"expected 2 labels, got {len(labels)}")
    pooled = np.vstack([X1, X2])
    first = halfspace_depth(X1, pooled, random_state=random_state)
    second = halfspace_depth(X2, pooled, random_state=random_state)
    ax = _axes(ax)
    ax.plot([0, 0.5], [0, 0.5], color=SECONDARY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    n1 = len(X1)
    _points(ax, first[:n1], second[:n1], BLUE, "o", f"{labels[0]} ({n1})")
    _points(ax, first[n1:], second[n1:], VIOLET, "^", f"{labels[1]} ({len(X2)})")
    ax.set(xlim=(-0.02, 0.52), ylim=(-0.02, 0.52), aspect="equal")
    ax.set_xlabel(f"Depth in {labels[0]}")
    ax.set_ylabel(f"Depth in {labels[1]}")
    _title(ax, "Depth-depth plot", "halfspace depth; points near the diagonal if one distribution")
    _legend(ax)
    return ax


def ctl_curves(
    X,
    *,
    n_clusters=(1, 2, 3, 4),
    alphas=(0.0, 0.04, 0.08, 0.12, 0.16, 0.2),
    restr_factor=50.0,
    n_init=500,
    random_state=None,
    ax=None,
):
    """Classification trimmed likelihood curves for choosing ``n_clusters`` and ``alpha``.

    Fits :class:`~robustmodels.cluster.TClust` for each number of clusters
    ``k`` and trimming level ``alpha`` and draws the trimmed classification
    log-likelihood against ``alpha``, one curve per ``k``, as R's
    ``ctlcurves`` (García-Escudero, Gordaliza, Matrán & Mayo-Iscar 2011). The
    likelihood always grows with ``k``; choose the smallest ``k`` whose curve
    is close to that of ``k + 1``, and the smallest ``alpha`` from which they
    stay close: a larger gap means an extra cluster, possibly made of outliers
    that more trimming would remove.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_clusters : sequence of int, default (1, 2, 3, 4)
    alphas : sequence of float in [0, 1), default six values from 0 to 0.2
    restr_factor : float, default 50.0
        Eigenvalue ratio of every fit, R's default for these curves.
    n_init : int, default 500
        Random starts of every fit; fewer are faster and less reliable.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    García-Escudero, L. A., Gordaliza, A., Matrán, C. and Mayo-Iscar, A.
    (2011). Exploring the number of groups in robust model-based clustering.
    *Statistics and Computing*, 21(4), 585-599.
    """
    from .cluster import TClust

    alphas = np.sort(np.asarray(alphas, dtype=float))
    ax = _axes(ax)
    for k, clusters in enumerate(n_clusters):
        objectives = [
            TClust(
                clusters, alpha=alpha, restr_factor=restr_factor, n_init=n_init,
                random_state=random_state,
            ).fit(X).objective_
            for alpha in alphas
        ]  # fmt: skip
        color, marker = _CLASSES[k % len(_CLASSES)]
        ax.plot(alphas, objectives, color=color, lw=1.4, marker=marker, ms=6, mec="white",
                mew=0.7, zorder=2, label=f"k = {clusters}")  # fmt: skip
    ax.set_xlabel(r"Trimming level $\alpha$")
    ax.set_ylabel("Trimmed classification log-likelihood")
    _title(ax, "Classification trimmed likelihood curves",
           f"TCLUST, eigenvalue ratio <= {restr_factor:g}")  # fmt: skip
    _legend(ax)
    return ax


def _least_squares(X, y):
    """Least squares estimates and classical standard errors, intercept first."""
    X = np.asarray(X, dtype=float)
    X = check_array(X[:, None] if X.ndim == 1 else X, dtype=float)
    y = as_1d(y)
    if len(y) != len(X):
        raise ValueError(f"y has {len(y)} values for {len(X)} rows of X")
    design = np.column_stack([np.ones(len(X)), X])
    estimates, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if len(y) <= rank or rank < design.shape[1]:
        raise ValueError(
            "least squares intervals require full rank and residual degrees of freedom"
        )
    residuals = y - design @ estimates
    variance = residuals @ residuals / (len(y) - design.shape[1])
    return estimates, np.sqrt(variance) * np.linalg.norm(np.linalg.pinv(design), axis=1)


def distance_qq(model, X, *, id_n=3, ax=None):
    """Chi-square quantile plot of robust distances.

    Sorted robust distances against the square roots of the chi-square
    quantiles at ``(i - 0.5) / n``. Regular multivariate normal observations
    follow the diagonal; outliers leave it at the upper end. The dashed line is
    the square root of the 0.975 chi-square quantile.

    Parameters
    ----------
    model : fitted covariance estimator with ``mahalanobis``
        For example :class:`~robustmodels.covariance.MCD`. ``ValueError`` if its
        covariance is singular, because the chi-square reference does not apply.
    X : array-like of shape (n_samples, n_features)
    id_n : int, default 3
        Positive maximum number of flagged distances labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Rousseeuw, P. J. and van Zomeren, B. C. (1990). Unmasking multivariate
    outliers and leverage points. *Journal of the American Statistical
    Association*, 85(411), 633-639.
    """
    check_count("id_n", id_n)
    X = np.asarray(X, dtype=float)
    _check_full_rank(model)
    p = X.shape[1]
    distances = np.sqrt(model.mahalanobis(X))
    quantiles = np.sqrt(chi2.ppf(_plotting_positions(distances), p))
    cut = np.sqrt(chi2.ppf(0.975, p))
    ax = _qq(ax, quantiles, distances, distances > cut, distances, id_n)
    _cutoff(ax, cut, "h", rf"$\sqrt{{\chi^2_{{{p},0.975}}}} = {cut:.2f}$")
    ax.set_xlabel(rf"Quantiles of $\sqrt{{\chi^2_{{{p}}}}}$")
    ax.set_ylabel(f"Robust distance ({type(model).__name__})")
    _title(ax, "Chi-square quantile plot", "regular normal observations follow the diagonal")
    _legend(ax)
    return ax


def residual_qq(model, X, y, *, id_n=3, ax=None):
    """Normal quantile plot of the standardized residuals of a robust regression.

    Sorted residuals ``(y - model.predict(X)) / model.scale_`` against the
    normal quantiles at ``(i - 0.5) / n``. Regular observations with normal
    errors follow the diagonal; the dashed lines mark +-2.5.

    Parameters
    ----------
    model : fitted regressor with ``predict`` and ``scale_``
        For example :class:`~robustmodels.regression.MMRegressor`.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        One response per row of ``X``.
    id_n : int, default 3
        Positive maximum number of flagged residuals labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    from scipy.stats import norm

    check_count("id_n", id_n)
    _, residuals = _standardized_residuals(model, X, y)
    quantiles = norm.ppf(_plotting_positions(residuals))
    ax = _qq(ax, quantiles, residuals, np.abs(residuals) > 2.5, np.abs(residuals), id_n)
    for value in (-2.5, 2.5):
        _cutoff(ax, value, "h", r"$\pm$2.5" if value > 0 else None)
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel(r"Standardized residual  $r_i / \hat\sigma$")
    _title(ax, "Normal quantile plot of residuals", type(model).__name__)
    _legend(ax)
    return ax


def residual_plot(model, X, y, *, id_n=3, ax=None):
    """Standardized residuals of a robust regression against its fitted values.

    Residuals ``(y - model.predict(X)) / model.scale_`` beyond +-2.5 are
    flagged. Because the fit is robust, outliers keep large residuals instead
    of pulling the fit towards them. A trend or a funnel shape among the
    regular observations suggests a missing term or a variance that changes
    with the mean.

    Parameters
    ----------
    model : fitted regressor with ``predict`` and ``scale_``
        For example :class:`~robustmodels.regression.MMRegressor`.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        One response per row of ``X``.
    id_n : int, default 3
        Positive maximum number of flagged residuals labeled with their
        zero-based row positions.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    References
    ----------
    Rousseeuw, P. J. and Leroy, A. M. (1987). *Robust Regression and Outlier
    Detection*. Wiley.
    """
    check_count("id_n", id_n)
    X, residuals = _standardized_residuals(model, X, y)
    fitted = model.predict(X)
    flagged = np.abs(residuals) > 2.5
    ax = _axes(ax)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    _points(
        ax, fitted[~flagged], residuals[~flagged], REGULAR, "o", f"Regular ({(~flagged).sum()})"
    )
    _points(ax, fitted[flagged], residuals[flagged], ORANGE, "D", f"Flagged ({flagged.sum()})")
    for value in (-2.5, 2.5):
        _cutoff(ax, value, "h", r"$\pm$2.5" if value > 0 else None)
    ax.set_ylim(min(residuals.min() * 1.15, -3.5), max(residuals.max() * 1.15, 3.5))
    _label_extremes(ax, fitted, residuals, np.abs(residuals), id_n, flagged)
    ax.set_xlabel("Fitted value")
    ax.set_ylabel(r"Standardized residual  $r_i / \hat\sigma$")
    _title(ax, "Residuals against fitted values", type(model).__name__)
    _legend(ax)
    return ax


def _standardized_residuals(model, X, y):
    """Two-dimensional ``X`` and the residuals of ``model`` divided by its scale."""
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    y = as_1d(y)
    if len(y) != len(X):
        raise ValueError(f"y has {len(y)} values for {len(X)} rows of X")
    if not model.scale_ > 0:
        raise ValueError("the model's scale_ is 0 (exact fit): residuals cannot be standardized")
    return X, (y - model.predict(X)) / model.scale_


def _plotting_positions(values):
    """Probability ``(rank - 0.5) / n`` of each value, in the order given."""
    return (rankdata(values, method="ordinal") - 0.5) / len(values)


def _qq(ax, quantiles, values, flagged, severity, id_n):
    """Points of a quantile plot with its diagonal; flagged points in orange."""
    ax = _axes(ax)
    ends = [quantiles.min(), quantiles.max()]
    ax.plot(ends, ends, color=AXIS, lw=0.9, zorder=1)
    _points(
        ax, quantiles[~flagged], values[~flagged], REGULAR, "o", f"Regular ({(~flagged).sum()})"
    )
    _points(ax, quantiles[flagged], values[flagged], ORANGE, "D", f"Flagged ({flagged.sum()})")
    _label_extremes(ax, quantiles, values, severity, id_n, flagged)
    return ax


def _check_full_rank(model):
    """The chi-square cutoffs assume a covariance matrix of full rank."""
    if _logdet(model.covariance_) == -np.inf:
        raise ValueError(
            f"the {type(model).__name__} covariance is singular: distances ignore the "
            "directions without spread, so the chi-square cutoff does not apply"
        )


def _axes(ax):
    """New axes with the package style, or ``ax`` restyled."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=_FIGSIZE, layout="constrained", facecolor="white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=SECONDARY, labelsize=8.5, width=0.8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set(color=INK, fontsize=9.5)
    ax.yaxis.label.set(color=INK, fontsize=9.5)
    return ax


def _points(ax, x, y, color, marker, label):
    """Scatter with a white ring around each marker; empty groups are skipped."""
    if len(x) == 0:
        return
    ax.scatter(
        x, y, s=30, color=color, marker=marker, edgecolors="white", linewidths=0.7,
        label=label, zorder=3,
    )  # fmt: skip


def _cutoff(ax, value, orientation, label):
    """Dashed reference line, labeled at its end in secondary ink."""
    style = {"color": SECONDARY, "lw": 0.9, "ls": (0, (4, 3)), "zorder": 1.5}
    if orientation == "h":
        ax.axhline(value, **style)
        if label:
            ax.annotate(
                label, (1, value), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=SECONDARY,
            )  # fmt: skip
    else:
        ax.axvline(value, **style)
        if label:
            ax.annotate(
                label, (value, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", fontsize=8, color=SECONDARY,
            )  # fmt: skip


def _label_extremes(ax, x, y, severity, id_n, eligible):
    """Label extreme indices, staggering nearby labels in display coordinates."""
    from matplotlib.transforms import Bbox

    ax.autoscale_view()
    candidates = np.flatnonzero(eligible)
    occupied = []
    for i in candidates[np.argsort(-severity[candidates], kind="stable")][:id_n]:
        # Work in points so spacing does not depend on data units or save DPI.
        px, py = ax.transData.transform((x[i], y[i])) * 72 / ax.figure.dpi
        width = 5 * len(str(i)) + 2
        step = 0
        while True:
            # Try both sides before increasing the vertical separation.
            dx = 5 if step % 2 == 0 else -5 - width
            dy = 4 + 12 * (step // 2)
            box = Bbox.from_bounds(px + dx, py + dy - 2, width, 10)
            if not any(box.overlaps(previous) for previous in occupied):
                break
            step += 1
        occupied.append(box)
        ax.annotate(
            str(i), (x[i], y[i]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6}
            if step else None,
        )  # fmt: skip


def _title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=16)
    ax.annotate(
        subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
        fontsize=8.5, color=SECONDARY, va="bottom",
    )  # fmt: skip


def _legend(ax, below=False):
    """Legend inside the axes, or in rows under the axis label when the data fill the axes."""
    if not below:
        ax.legend(
            frameon=False, fontsize=8, loc="best", handletextpad=0.4, borderaxespad=0.3,
            labelcolor=SECONDARY,
        )  # fmt: skip
        return
    entries = len(ax.get_legend_handles_labels()[1])
    ax.legend(
        frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17),
        ncol=min(entries, 3), handletextpad=0.4, columnspacing=1.2, labelcolor=SECONDARY,
    )  # fmt: skip


def _ellipse(center, covariance, radius, points=200):
    """Boundary of ``{x : (x - c)' S^-1 (x - c) = radius^2}``."""
    angle = np.linspace(0, 2 * np.pi, points)
    values, vectors = np.linalg.eigh(covariance)
    circle = np.vstack([np.cos(angle), np.sin(angle)])
    boundary = vectors @ (np.sqrt(values)[:, None] * circle) * radius
    return center[0] + boundary[0], center[1] + boundary[1]


def _fivenum(x):
    """Tukey's five-number summary, as R's ``fivenum``."""
    x = np.sort(x)
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    depth = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n]) - 1
    return 0.5 * (x[np.floor(depth).astype(int)] + x[np.ceil(depth).astype(int)])


def _adjusted_box_stats(x):
    """Box statistics of the adjusted boxplot, as R's ``robustbase::adjboxStats``."""
    _, q1, median, q3, _ = _fivenum(x)
    mc = medcouple(x)
    lower, upper = _adjusted_fences(q1, q3, mc)
    out = (x < lower) | (x > upper)
    inside = x[~out]
    return {
        "med": median,
        "q1": q1,
        "q3": q3,
        "whislo": inside.min(),
        "whishi": inside.max(),
        "fliers": x[out],
        "mc": mc,
        "fences": (lower, upper),
    }
