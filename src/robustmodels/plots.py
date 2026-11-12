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
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance

from ._validation import as_1d, check_count
from .scale import medcouple

__all__ = [
    "adjusted_boxplot",
    "distance_distance",
    "outlier_map",
    "outlyingness_plot",
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
    y : array-like of shape (n_samples,)
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
    from .covariance import MCD

    check_count("id_n", id_n)
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    if not model.scale_ > 0:
        raise ValueError("the model's scale_ is 0 (exact fit): residuals cannot be standardized")
    residuals = (as_1d(y) - model.predict(X)) / model.scale_
    distances = np.sqrt(MCD(random_state=random_state).fit(X).dist_)
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
        For example :class:`~robustmodels.covariance.MCD`.
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
    robust = np.sqrt(model.mahalanobis(X))
    # EmpiricalCovariance uses divisor n; classical diagnostics use n - 1.
    classical = np.sqrt(EmpiricalCovariance().fit(X).mahalanobis(X) * (len(X) - 1) / len(X))
    cut = np.sqrt(chi2.ppf(0.975, X.shape[1]))
    flagged = robust > cut
    ax = _axes(ax)
    top = 1.08 * max(robust.max(), classical.max(), cut * 1.25)
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
    _legend(ax)
    return ax


def adjusted_boxplot(x, *, labels=None, ax=None):
    """Skewness-adjusted boxplot (Hubert & Vandervieren 2008).

    The whiskers reach the most extreme observations inside the fences
    ``Q1 - 1.5 exp(-4 MC) IQR`` and ``Q3 + 1.5 exp(3 MC) IQR`` for medcouple
    ``MC >= 0`` (``exp(-3 MC)`` and ``exp(4 MC)`` otherwise); observations
    beyond them are drawn individually. The box spans Tukey's hinges, as R's
    ``robustbase::adjbox``. Each box is annotated with its medcouple.

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
        [{**s, "label": name} for s, name in zip(stats, names, strict=True)],
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
    for position, s in enumerate(stats, start=1):
        ax.annotate(
            f"MC = {s['mc']:.2f}",
            (position, s["whishi"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=SECONDARY,
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
    detector : fitted :class:`~robustmodels.outliers.StahelDonohoOutlyingness` or
        :class:`~robustmodels.outliers.AdjustedOutlyingness`
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
    name = type(detector).__name__
    measure = (
        "Adjusted outlyingness" if name.startswith("Adjusted") else "Stahel-Donoho outlyingness"
    )
    ax.set_ylabel(measure)
    _title(ax, measure, f"{flagged.sum()} of {len(values)} observations above the cutoff")
    _legend(ax)
    return ax


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


def _legend(ax):
    ax.legend(
        frameon=False, fontsize=8, loc="best", handletextpad=0.4, borderaxespad=0.3,
        labelcolor=SECONDARY,
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
    iqr = q3 - q1
    mc = medcouple(x)
    if mc >= 0:
        fences = q1 - 1.5 * np.exp(-4 * mc) * iqr, q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        fences = q1 - 1.5 * np.exp(-3 * mc) * iqr, q3 + 1.5 * np.exp(4 * mc) * iqr
    out = (x < fences[0]) | (x > fences[1])
    inside = x[~out]
    return {
        "med": median,
        "q1": q1,
        "q3": q3,
        "whislo": inside.min(),
        "whishi": inside.max(),
        "fliers": x[out],
        "mc": mc,
        "fences": fences,
    }
