"""Halfspace depth, regression depth and the elements of the bagplot.

Depths are returned as fractions of the sample size, as R's mrfDepth: a depth
of ``k / n`` means that ``k`` observations lie on the least favourable side.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array, check_X_y

from ._validation import as_generator, check_count, check_range
from .outliers import _directions, _merge_ties

__all__ = ["bagplot_stats", "halfspace_depth", "halfspace_median", "regression_depth"]

_ANGLE_TOL = 1e-12  # angles closer than this are the same direction
_TIE_TOL = 1e-10  # projections closer than this (relative to their range) are tied


def halfspace_depth(X, points=None, *, n_directions=None, random_state=None):
    """Halfspace (Tukey) depth of ``points`` relative to the sample ``X``.

    The depth of ``z`` is the smallest fraction of observations in a closed
    halfspace whose boundary passes through ``z`` (Tukey 1975). It is exact for
    one and two variables (the latter as Rousseeuw & Ruts 1996, in
    ``O(n log n)`` per point). For more variables it is the smallest univariate
    depth over directions normal to hyperplanes through ``p`` observations,
    which is affine invariant and never below the exact depth, as the
    approximate algorithm of R's mrfDepth.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample.
    points : array-like of shape (n_points, n_features), optional
        Points whose depth is computed; the rows of ``X`` by default.
    n_directions : int, default None
        Directions for three or more variables; ``250 * n_features`` by default.
        All hyperplanes are used when there are at most that many.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions.

    Returns
    -------
    ndarray of shape (n_points,)
        Depths in [0, 1]. In one and two dimensions, points outside the convex
        hull have depth 0; approximate depths in higher dimensions may be positive.

    References
    ----------
    Tukey, J. W. (1975). Mathematics and the picturing of data. *Proceedings of
    the International Congress of Mathematicians*, 2, 523-531.

    Rousseeuw, P. J. and Ruts, I. (1996). Algorithm AS 307: Bivariate location
    depth. *Journal of the Royal Statistical Society: Series C*, 45(4), 516-526.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.depth import halfspace_depth
    >>> square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    >>> halfspace_depth(square).tolist()
    [0.2, 0.2, 0.2, 0.2, 0.6]
    >>> halfspace_depth(square, [[2.0, 2.0]]).tolist()
    [0.0]
    """
    X = check_array(X, dtype=np.float64, ensure_min_samples=2)
    points = X if points is None else check_array(points, dtype=np.float64)
    n, p = X.shape
    if points.shape[1] != p:
        raise ValueError(f"points has {points.shape[1]} features, X has {p}")
    if n_directions is not None:
        check_count("n_directions", n_directions)
    if p == 2:
        return np.array([_depth_2d(z, X) for z in points]) / n
    n_directions = 250 * p if n_directions is None else n_directions
    directions = _directions(X, n_directions, as_generator(random_state))
    return _projected_depth(X @ directions, points @ directions) / n


def _depth_2d(z, X):
    """Number of observations in the least populated closed halfplane through ``z``."""
    differences = X - z
    away = differences.any(axis=1)
    angles = np.sort(np.arctan2(differences[away, 1], differences[away, 0]))
    if not angles.size:
        return len(X)
    # A halfplane in general position is the arc (start, start + pi] that starts
    # just after an observation or just after the opposite of one.
    circle = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
    starts = np.concatenate([angles, angles - np.pi]) + _ANGLE_TOL
    inside = np.searchsorted(circle, starts + np.pi, "right") - np.searchsorted(
        circle, starts, "right"
    )
    return len(X) - int(away.sum()) + int(inside.min())


def _projected_depth(projections, points):
    """Smallest univariate depth (as a count) over the columns of the projections."""
    n = len(projections)
    depth = np.full(len(points), n)
    for column, values in zip(np.sort(projections, axis=0).T, points.T, strict=True):
        tol = _TIE_TOL * (column[-1] - column[0])
        below = np.searchsorted(column, values + tol, "right")
        above = n - np.searchsorted(column, values - tol, "left")
        depth = np.minimum(depth, np.minimum(below, above))
    return depth


def regression_depth(X, y, coef, intercept=0.0, *, n_directions=None, random_state=None):
    """Regression depth of the fit ``y = intercept + X @ coef`` (Rousseeuw & Hubert 1999).

    The regression depth is the smallest fraction of observations that must be
    removed to make the fit a nonfit: a fit that can be tilted around some
    hyperplane of the ``X`` space until it is vertical without passing through
    any observation. A fit with depth 0 fits no data; the deepest fits have a
    depth of at least ``1 / (n_features + 2)``, and close to 1/2 when the
    linear model holds with symmetric errors. The depth is exact for one
    regressor. For more, the tilting hyperplanes pass through ``n_features``
    observations: all of them when there are at most ``n_directions``, which
    is exact for data in general position, otherwise a random selection, which
    never underestimates the depth.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Regressors, without a constant column.
    y : array-like of shape (n_samples,)
        Response.
    coef : array-like of shape (n_features,)
        Slopes of the fit.
    intercept : float, default 0.0
        Intercept of the fit.
    n_directions : int, default None
        Number of tilting directions; ``250 * n_features`` by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions.

    Returns
    -------
    float
        Regression depth in [0, 1].

    References
    ----------
    Rousseeuw, P. J. and Hubert, M. (1999). Regression depth. *Journal of the
    American Statistical Association*, 94(446), 388-402.

    Rousseeuw, P. J. and Struyf, A. (1998). Computing location depth and
    regression depth in higher dimensions. *Statistics and Computing*, 8, 193-203.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.depth import regression_depth
    >>> x = np.arange(10.0)
    >>> y = 1 + 2 * x + np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1.0])
    >>> regression_depth(x[:, None], y, [2.0], 1.0)  # residual signs alternate
    0.4
    >>> regression_depth(x[:, None], y, [2.0], 5.0)  # the line lies above the data
    0.0
    """
    X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, ensure_min_samples=2)
    n, p = X.shape
    coef = np.asarray(coef, dtype=float).reshape(-1)
    if not np.isfinite(coef).all() or not np.isscalar(intercept) or not np.isfinite(intercept):
        raise ValueError("coef and intercept must be finite")
    if coef.size != p:
        raise ValueError(f"coef has {coef.size} entries, X has {p} features")
    if n_directions is not None:
        check_count("n_directions", n_directions)
    n_directions = 250 * p if n_directions is None else n_directions
    residuals = y - X @ coef - intercept
    residuals[np.abs(residuals) <= _TIE_TOL * np.ptp(y)] = 0.0
    directions = _directions(X, n_directions, as_generator(random_state))
    projections = _merge_ties(X @ directions)
    return float(min(_tilting_depth(t, residuals, p) for t in projections.T) / n)


def _tilting_depth(t, residuals, p):
    """Fewest observations to remove so that the fit can be tilted around some ``t = v``.

    Tilting around ``v`` needs residuals of one sign on its left and of the
    other sign on its right. ``v`` lies between tied groups of ``t`` or on a
    group of at most ``p`` observations, which then fall on either side (a
    hyperplane through ``p`` observations in general position can be moved to
    leave each on the side that suits it).
    """
    order = np.argsort(t, kind="stable")
    t, residuals = t[order], residuals[order]
    positive = np.r_[0, np.cumsum(residuals >= 0)]
    negative = np.r_[0, np.cumsum(residuals <= 0)]
    zero = np.r_[0, np.cumsum(residuals == 0)]
    starts = np.flatnonzero(np.r_[True, np.diff(t) > 0])
    ends = np.r_[starts[1:], len(t)]

    def removed(left, right):  # left: observations before v, right: from index right on
        up = positive[left] + negative[-1] - negative[right]
        down = negative[left] + positive[-1] - positive[right]
        return np.minimum(up, down)

    cuts = np.r_[starts, len(t)]
    between = removed(cuts, cuts).min()
    small = ends - starts <= p
    through = removed(starts[small], ends[small]) + zero[ends[small]] - zero[starts[small]]
    return min(between, through.min(initial=between))


def halfspace_median(X, *, max_samples=200, random_state=None):
    """Tukey median of bivariate data: the centre of gravity of the deepest region.

    The deepest region is the set of points with the largest halfspace depth, a
    convex polygon (or a segment or a point). The Tukey median is affine
    equivariant and its breakdown point is at least 1/3.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Sample.
    max_samples : int, default 200
        The depth regions are computed from a random subset of this size when
        the sample is larger, as mrfDepth; the search is exact otherwise.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the subset.

    Returns
    -------
    ndarray of shape (2,)

    References
    ----------
    Rousseeuw, P. J. and Ruts, I. (1998). Constructing the bivariate Tukey
    median. *Statistica Sinica*, 8(3), 827-839.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.depth import halfspace_median
    >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> halfspace_median(X).round(6).tolist()
    [0.5, 0.5]
    """
    return _Regions(X, max_samples, random_state).median()


def bagplot_stats(X, *, factor=3.0, max_samples=200, random_state=None):
    """Elements of the bagplot, the bivariate boxplot of Rousseeuw, Ruts & Tukey (1999).

    The bag contains half of the observations: it lies between the two
    consecutive halfspace depth regions that hold at most and more than half of
    them, interpolated along rays from the Tukey median. The fence is the bag
    inflated by ``factor`` from the median; observations outside it are
    outliers, and the loop is the convex hull of the bag and the others.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Sample.
    factor : float, default 3.0
        Inflation of the bag that gives the fence.
    max_samples : int, default 200
        The median and the bag are computed from a random subset of this size
        when the sample is larger, as mrfDepth.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the subset.

    Returns
    -------
    sklearn.utils.Bunch
        ``center`` (Tukey median), ``bag``, ``fence`` and ``loop`` (polygon
        vertices in counterclockwise order, one per row), ``outliers`` (boolean
        mask of the observations outside the fence) and ``depth`` (halfspace
        depth of each observation).

    References
    ----------
    Rousseeuw, P. J., Ruts, I. and Tukey, J. W. (1999). The bagplot: a bivariate
    boxplot. *The American Statistician*, 53(4), 382-387.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.depth import bagplot_stats
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> X[:3] = [[6, 6], [7, -6], [-8, 5]]
    >>> np.flatnonzero(bagplot_stats(X).outliers).tolist()
    [0, 1, 2]
    """
    check_range("factor", factor, 1, np.inf, high_open=True)
    regions = _Regions(X, max_samples, random_state)
    center = regions.median()
    bag = _hull(regions.bag(center))
    fence = center + factor * (bag - center)
    outliers = ~_inside(regions.X, center, fence)
    loop = _hull(np.vstack([bag, regions.X[~outliers]]))
    depth = halfspace_depth(regions.X)
    return Bunch(center=center, bag=bag, fence=fence, loop=loop, outliers=outliers, depth=depth)


class _Regions:
    """Halfspace depth regions of bivariate data.

    The region of depth ``k`` is the intersection of the halfplanes
    ``u'x >= k-th smallest projection on u`` over all directions ``u``. The
    order of the projections only changes at the normals of the lines through
    two observations, so those directions (and their opposites) are enough.
    Computations use a standardized subset; results are mapped back.
    """

    def __init__(self, X, max_samples, random_state):
        check_count("max_samples", max_samples)
        if max_samples < 3:
            raise ValueError("max_samples must be at least 3")
        self.X = X = check_array(X, dtype=np.float64, ensure_min_samples=3)
        if X.shape[1] != 2:
            raise ValueError(f"expected bivariate data, got {X.shape[1]} features")
        self.shift, self.scale = np.median(X, axis=0), np.ptp(X, axis=0)
        if np.linalg.matrix_rank(X - X[0]) < 2:
            raise ValueError("X is singular: the observations lie on a line")
        Z = (X - self.shift) / self.scale
        if len(Z) > max_samples:
            Z = Z[as_generator(random_state).choice(len(Z), max_samples, replace=False)]
            if np.linalg.matrix_rank(Z - Z[0]) < 2:
                raise ValueError("the selected subset is singular; increase max_samples")
        i, j = np.triu_indices(len(Z), 1)
        normals = (Z[j] - Z[i])[:, ::-1] * [1, -1]
        lengths = np.linalg.norm(normals, axis=1)
        self.normals = normals[lengths > 0] / lengths[lengths > 0, None]
        self.sorted = np.sort(Z @ self.normals.T, axis=0)
        self.counts = np.round(halfspace_depth(Z) * len(Z)).astype(int)

    def _halfplanes(self, k):
        """``A, b`` with the region of depth ``k`` equal to ``{x: A x <= b}``."""
        A = np.vstack([-self.normals, self.normals])
        return A, np.r_[-self.sorted[k - 1], self.sorted[-k]]

    def _chebyshev(self, k):
        """Centre and radius of the largest disc in the region; radius < 0 if it is empty."""
        A, b = self._halfplanes(k)
        fit = linprog([0, 0, -1], np.c_[A, np.ones(len(A))], b, bounds=[(None, None)] * 3)
        return (fit.x[:2], fit.x[2]) if fit.status == 0 else (None, -1.0)

    def polygon(self, k):
        """Vertices of the region of depth ``k`` (standardized), or None if it is empty."""
        center, radius = self._chebyshev(k)
        if radius < -1e-9:
            return None
        A, b = self._halfplanes(k)
        if radius > 1e-9:
            return _hull(HalfspaceIntersection(np.c_[A, -b], center).intersections)
        # A segment or a point: its ends are the extremes along the two axes.
        ends = [linprog(c, A, b, bounds=[(None, None)] * 2).x for c in ([1, 0], [-1, 0], [0, 1])]
        return _hull(np.unique(np.round(ends, 9), axis=0))

    def _deepest(self):
        """Largest depth with a nonempty region; it is at least the largest data depth."""
        low, high = self.counts.max(), len(self.counts) // 2 + 1
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (middle, high) if self._chebyshev(middle)[1] >= -1e-9 else (low, middle)
        return low

    def _restore(self, points):
        return points * self.scale + self.shift

    def median(self):
        return self._restore(_centroid(self.polygon(self._deepest())))

    def bag(self, center):
        """Interpolate, along rays from the centre, the two regions around half the data."""
        n = len(self.counts)
        center = (center - self.shift) / self.scale
        held = np.array([(self.counts >= k).sum() for k in range(n + 2)])
        k = int(np.argmax(held <= n // 2))  # held[k] <= n // 2 < held[k - 1]
        outer, inner = self.polygon(k - 1), self.polygon(k)
        angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        if inner is not None and len(inner) > 2:
            angles = np.r_[angles, _angles(inner, center)]
        angles = np.unique(np.r_[angles, _angles(outer, center)])
        share = (n / 2 - held[k]) / (held[k - 1] - held[k])
        radius = share * _radius(outer, center, angles)
        if inner is not None and len(inner) > 2:
            radius += (1 - share) * _radius(inner, center, angles)
        rays = np.c_[np.cos(angles), np.sin(angles)]
        return self._restore(center + radius[:, None] * rays)


def _hull(points):
    """Vertices of the convex hull, counterclockwise."""
    if len(points) < 3:
        return points
    if np.linalg.matrix_rank(points - points[0]) < 2:
        axis = np.argmax(np.ptp(points, axis=0))
        return np.unique(points[[np.argmin(points[:, axis]), np.argmax(points[:, axis])]], axis=0)
    return points[ConvexHull(points).vertices]


def _centroid(polygon):
    """Centre of gravity of a convex polygon, a segment or a point."""
    if len(polygon) < 3:
        return polygon.mean(axis=0)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    return ((polygon + following) * cross[:, None]).sum(axis=0) / (3 * cross.sum())


def _angles(points, center):
    return np.arctan2(*(points - center).T[::-1]) % (2 * np.pi)


def _radius(polygon, center, angles):
    """Distance from ``center`` (inside the convex polygon) to its boundary along each angle."""
    if len(polygon) < 3:
        rays = np.c_[np.cos(angles), np.sin(angles)]
        offsets = polygon - center
        cross = rays[:, 0, None] * offsets[:, 1] - rays[:, 1, None] * offsets[:, 0]
        tolerance = 1e-10 * max(np.linalg.norm(offsets, axis=1).max(), np.finfo(float).eps)
        aligned = np.abs(cross).max(axis=1) <= tolerance
        return np.where(aligned, np.maximum((rays @ offsets.T).max(axis=1), 0), 0)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = edges[:, ::-1] * [1, -1]  # outward for a counterclockwise polygon
    slack = np.maximum(np.sum(normals * (polygon - center), axis=1), 0)
    speed = np.c_[np.cos(angles), np.sin(angles)] @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(speed > 1e-14, slack / speed, np.inf).min(axis=1)


def _inside(points, center, polygon):
    """Whether each point lies in the convex polygon, which contains ``center``."""
    offsets = points - center
    reach = _radius(polygon, center, np.arctan2(offsets[:, 1], offsets[:, 0]))
    return np.linalg.norm(offsets, axis=1) <= reach * (1 + 1e-9)
