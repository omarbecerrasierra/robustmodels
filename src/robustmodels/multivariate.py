"""Robust multivariate methods: principal components, discriminant analysis and tests.

The principal component estimators are scikit-learn transformers: ``fit(X)``
sets ``location_``, ``components_`` and ``explained_variance_``, and
``transform(X)`` returns the scores. The discriminant rules are scikit-learn
classifiers. :func:`hotelling_t2` and :func:`manova` test hypotheses on mean
vectors and return named tuples.
"""

from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.stats import chi2
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_generator, check_count, check_option, check_range
from .covariance import (
    _SCALES,
    MCD,
    _logdet,
    _mahalanobis_sq,
    _univariate_mcd,
    h_alpha_n,
)

__all__ = [
    "ROBPCA",
    "ProjectionPursuitPCA",
    "RobustLDA",
    "RobustQDA",
    "hotelling_t2",
    "manova",
]

_Z975 = stats.norm.ppf(0.975)


class _RobustPCA(TransformerMixin, BaseEstimator):
    """Base class: scores, reconstruction and the two outlier distances."""

    def transform(self, X):
        """Scores of ``X``: coordinates on the principal components."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return (X - self.location_) @ self.components_.T

    def inverse_transform(self, X):
        """Points of the original space with scores ``X``."""
        check_is_fitted(self)
        return np.asarray(X, dtype=float) @ self.components_ + self.location_

    def distances(self, X):
        """Score and orthogonal distances of each row of ``X``.

        The score distance is the robust distance of the scores inside the
        principal subspace; the orthogonal distance is the Euclidean distance
        from the observation to that subspace.

        Returns
        -------
        score_distances, orthogonal_distances : ndarray of shape (n_samples,)
        """
        scores = self.transform(X)
        score = np.sqrt(np.sum(scores**2 / self.explained_variance_, axis=1))
        residuals = np.asarray(X, dtype=float) - self.inverse_transform(scores)
        return score, np.linalg.norm(residuals, axis=1)

    def _set_components(self, location, components, variances, X, rank):
        largest = np.abs(components).argmax(axis=1)
        signs = np.sign(components[np.arange(len(components)), largest])
        self.location_ = location
        self.components_ = components * signs[:, None]
        self.explained_variance_ = variances
        self.n_components_ = len(components)
        self.score_distances_, self.orthogonal_distances_ = self.distances(X)
        self.score_cutoff_ = float(np.sqrt(chi2.ppf(0.975, self.n_components_)))
        self.orthogonal_cutoff_ = 0.0
        if self.n_components_ < rank:
            self.orthogonal_cutoff_ = _orthogonal_cutoff(self.orthogonal_distances_, self.h_)


class ROBPCA(_RobustPCA):
    """Robust principal components by ROBPCA (Hubert, Rousseeuw & Vanden Branden 2005).

    The data are first reduced, without loss, to the subspace they span, so
    the method works with more variables than observations. Each observation
    then receives a Stahel-Donoho outlyingness: the largest
    ``|v'x - t(v'X)| / s(v'X)`` over directions ``v`` through two observations,
    with ``t`` and ``s`` the reweighted univariate MCD location and scale. The
    ``h`` least outlying observations give a first subspace, the observations
    close to it give the final ``k``-dimensional subspace, and the reweighted
    :class:`~robustmodels.covariance.MCD` of the data projected on it gives the
    center, the components and their variances. When the data have few
    dimensions (``rank <= min(n // 5, max_components)``) the MCD is applied
    directly and its eigenvectors are the components.

    The score distance of an observation measures how far its scores are from
    the center, and its orthogonal distance how far it lies from the subspace.
    Observations beyond ``score_cutoff_`` or ``orthogonal_cutoff_`` are
    outlying; :func:`~robustmodels.plots.pca_outlier_map` shows both.

    Parameters
    ----------
    n_components : int, default None
        Number of components ``k``. By default, the smallest number that
        explains 80% of the variance of the ``h`` least outlying observations,
        among the components with a variance above 0.001 times the largest,
        and at most ``max_components``.
    alpha : float in [0.5, 1], default 0.75
        Coverage: the method resists up to a fraction ``1 - alpha`` of outliers.
        The subset size is ``h_alpha_n(alpha, n, k_max)``, with ``k_max`` the
        smaller of ``max_components`` and the rank of the data.
    max_components : int, default 10
        Largest number of components chosen when ``n_components`` is None.
    n_directions : int, default 250
        Directions for the outlyingness; all pairs of observations are used
        when there are at most this many.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed for the random directions and for the MCD.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Robust center.
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in rows, by decreasing variance. The largest loading of
        each axis is positive.
    explained_variance_ : ndarray of shape (n_components,)
        Robust variance of the scores on each component.
    n_components_ : int
        Number of components.
    h_ : int
        Subset size.
    score_distances_, orthogonal_distances_ : ndarray of shape (n_samples,)
        Distances of the training data.
    score_cutoff_ : float
        Square root of the 0.975 chi-square quantile with ``n_components_``
        degrees of freedom.
    orthogonal_cutoff_ : float
        ``(m + s z)^(3/2)``, with ``m`` and ``s`` the univariate MCD location
        and scale of the orthogonal distances to the power 2/3 and ``z`` the
        0.975 normal quantile; 0 when the components span the data.
    support_ : ndarray of bool, shape (n_samples,)
        Observations within both cutoffs.

    Notes
    -----
    R's ``rrcov::PcaHubert`` estimates the orthogonal cutoff with the median
    and MAD instead of the univariate MCD of the paper, and may end with a
    reweighted covariance that has no consistency factor.

    References
    ----------
    Hubert, M., Rousseeuw, P. J. and Vanden Branden, K. (2005). ROBPCA: a new
    approach to robust principal component analysis. *Technometrics*, 47(1),
    64-79.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.multivariate import ROBPCA
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 20))  # a plane in 20-D
    >>> X += 0.1 * rng.normal(size=X.shape)
    >>> X[:10] += 5  # ten outliers away from the plane
    >>> pca = ROBPCA(n_components=2, random_state=0).fit(X)
    >>> pca.transform(X).shape
    (100, 2)
    >>> bool(np.all(pca.orthogonal_distances_[:10] > pca.orthogonal_cutoff_))
    True
    >>> int(pca.support_[10:].sum())  # regular observations within both cutoffs
    87
    """

    def __init__(
        self,
        n_components=None,
        *,
        alpha=0.75,
        max_components=10,
        n_directions=250,
        random_state=None,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.max_components = max_components
        self.n_directions = n_directions
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=3)
        check_range("alpha", self.alpha, 0.5, 1)
        for name in ("max_components", "n_directions"):
            check_count(name, getattr(self, name))
        if self.n_components is not None:
            check_count("n_components", self.n_components)
        rng = as_generator(self.random_state)
        mean, _, basis = _classical_pca(X)
        rank = basis.shape[1]
        if rank == 0:
            raise ValueError("all observations are identical")
        Z = (X - mean) @ basis  # the data in the subspace they span
        k_max = min(self.n_components or self.max_components, rank)
        self.h_ = h_alpha_n(self.alpha, len(X), min(self.max_components, rank))
        if rank <= min(len(X) // 5, self.max_components):
            center, axes, variances = self._mcd_components(Z, np.eye(rank), rng)
            k = k_max if self.n_components else _n_components(variances, k_max)
            axes, variances = axes[:, :k], variances[:k]
        else:
            origin, subspace = self._robust_subspace(Z, k_max, rng)
            center, axes, variances = self._mcd_components(Z - origin, subspace, rng)
            center += origin
        self._set_components(mean + center @ basis.T, (basis @ axes).T, variances, X, rank)
        within = self.orthogonal_distances_ <= self.orthogonal_cutoff_
        self.support_ = (self.score_distances_ <= self.score_cutoff_) & (
            within | (self.orthogonal_cutoff_ == 0)
        )
        return self

    def _mcd_components(self, Z, subspace, rng):
        """MCD center, eigenvectors and eigenvalues of ``Z`` projected on ``subspace``."""
        mcd = MCD(alpha=self.h_ / len(Z), random_state=rng).fit(Z @ subspace)
        variances, vectors = np.linalg.eigh(mcd.covariance_)
        return mcd.location_ @ subspace.T, subspace @ vectors[:, ::-1], variances[::-1]

    def _robust_subspace(self, Z, k_max, rng):
        """Center and orthonormal basis of the subspace of the least outlying observations."""
        h = self.h_
        outlyingness = _pair_outlyingness(Z, h, self.n_directions, rng)
        least = np.argsort(outlyingness, kind="stable")[:h]
        center, variances, vectors = _classical_pca(Z[least])
        k = min(k_max, len(variances))
        if self.n_components is None:
            k = _n_components(variances, k)
        if k < Z.shape[1]:  # refit on every observation close to that subspace
            centered = Z - center
            residuals = centered - centered @ vectors[:, :k] @ vectors[:, :k].T
            distances = np.linalg.norm(residuals, axis=1)
            close = distances <= _orthogonal_cutoff(distances, h)
            center, variances, vectors = _classical_pca(Z[close])
            k = min(k, len(variances))
        return center, vectors[:, :k]


class ProjectionPursuitPCA(_RobustPCA):
    """Robust principal components by projection pursuit (Li & Chen 1985).

    The first component is the direction that maximizes a robust scale of the
    projected data; each next one maximizes it among the directions orthogonal
    to the previous components. The search uses the grid algorithm of Croux,
    Filzmoser & Oliveira (2007): it improves the direction in the plane of the
    current direction and one variable at a time, over a grid of angles that is
    halved at every cycle. The result is a local optimum of each criterion.
    The data are centered at the spatial median. No covariance matrix is
    needed, so the method works with more variables than observations, and the
    components are orthogonally equivariant up to the grid resolution.

    Parameters
    ----------
    n_components : int, default 2
        Number of components.
    scale : {"mad", "qn", "tau", "pn"} or callable, default "mad"
        Robust scale to maximize, or a function of a one-dimensional sample.
        "qn" is more efficient than "mad" but much slower for large samples.
        With the standard deviation the method estimates classical components.
    n_grid : int, default 25
        Angles tried in each plane.
    n_cycles : int, default 10
        Cycles over the variables; the grid is halved at each cycle.
    alpha : float in [0.5, 1], default 0.75
        Coverage of the univariate MCD in the orthogonal distance cutoff.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Spatial median.
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in rows, in the order found. The largest loading of each
        axis is positive.
    explained_variance_ : ndarray of shape (n_components,)
        Squared robust scale of the scores on each component.
    n_components_ : int
        Number of components.
    score_distances_, orthogonal_distances_ : ndarray of shape (n_samples,)
        Distances of the training data.
    score_cutoff_, orthogonal_cutoff_ : float
        Cutoffs, as in :class:`ROBPCA`.

    References
    ----------
    Li, G. and Chen, Z. (1985). Projection-pursuit approach to robust dispersion
    matrices and principal components: primary theory and Monte Carlo. *Journal
    of the American Statistical Association*, 80(391), 759-766.

    Croux, C., Filzmoser, P. and Oliveira, M. R. (2007). Algorithms for
    projection-pursuit robust principal component analysis. *Chemometrics and
    Intelligent Laboratory Systems*, 87(2), 218-225.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.multivariate import ProjectionPursuitPCA
    >>> X = np.random.default_rng(0).normal(size=(100, 3)) * [5, 1, 1]
    >>> X[:10] = [0, 100, 0]  # ten outliers on the second axis
    >>> pca = ProjectionPursuitPCA(n_components=1).fit(X)
    >>> abs(pca.components_[0]).round(1).tolist()
    [1.0, 0.1, 0.1]
    """

    def __init__(self, n_components=2, *, scale="mad", n_grid=25, n_cycles=10, alpha=0.75):
        self.n_components = n_components
        self.scale = scale
        self.n_grid = n_grid
        self.n_cycles = n_cycles
        self.alpha = alpha

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=3)
        for name in ("n_components", "n_grid", "n_cycles"):
            check_count(name, getattr(self, name))
        check_range("alpha", self.alpha, 0.5, 1)
        if not callable(self.scale):
            check_option("scale", self.scale, _SCALES)
        scale_fn = self.scale if callable(self.scale) else _SCALES[self.scale]
        n, p = X.shape
        if self.n_components > min(n - 1, p):
            raise ValueError(
                f"n_components must be at most min(n - 1, p) = {min(n - 1, p)}, "
                f"got {self.n_components}"
            )
        self.h_ = h_alpha_n(self.alpha, n, self.n_components)
        location = _spatial_median(X)
        projected, basis = X - location, np.eye(p)
        if p >= n:  # search inside the subspace spanned by the observations
            basis = np.linalg.svd(projected, full_matrices=False)[2].T
            projected = projected @ basis
        components, variances = [], []
        for _ in range(self.n_components):
            direction = _grid_direction(projected, scale_fn, self.n_grid, self.n_cycles)
            components.append(basis @ direction)
            variances.append(scale_fn(projected @ direction) ** 2)
            complement = np.linalg.qr(direction[:, None], mode="complete")[0][:, 1:]
            projected, basis = projected @ complement, basis @ complement
        if min(variances) <= 0:
            raise ValueError("a component has zero robust scale; use fewer components")
        rank = np.linalg.matrix_rank(X - location)
        self._set_components(location, np.array(components), np.array(variances), X, rank)
        return self


def _classical_pca(X):
    """Mean, variances and loadings (in columns) of the components with nonzero variance."""
    center = X.mean(axis=0)
    _, singular, vt = np.linalg.svd(X - center, full_matrices=False)
    rank = int(np.sum(singular > max(X.shape) * np.finfo(float).eps * singular[0]))
    return center, singular[:rank] ** 2 / (len(X) - 1), vt[:rank].T


def _n_components(variances, k_max):
    """Components with variance above 0.001 times the largest, up to 80% of the total."""
    k = min(k_max, int(np.sum(np.cumprod(variances > 1e-3 * variances[0]))))
    explained = np.cumsum(variances[:k]) / variances.sum()
    return min(k, int(np.searchsorted(explained, 0.8)) + 1)


def _unimcd(y, h):
    """Reweighted univariate MCD location and scale, as in ROBPCA."""
    n = len(y)
    subset = y[_univariate_mcd(y, h)]
    center, variance = subset.mean(), subset.var(ddof=1)
    if h == n or variance == 0:
        return center, np.sqrt(variance)
    squared = (y - center) ** 2 / variance
    consistency = np.partition(squared, h - 1)[h - 1] / chi2.ppf(h / n, 1)
    kept = y[squared < consistency * chi2.ppf(0.975, 1)]
    return kept.mean(), kept.std(ddof=1)


def _pair_outlyingness(Z, h, n_directions, rng):
    """Stahel-Donoho outlyingness over directions through two observations."""
    n = len(Z)
    if n * (n - 1) // 2 <= n_directions:
        first, second = np.triu_indices(n, 1)
    else:
        first = rng.integers(n, size=n_directions)
        second = (first + rng.integers(1, n, size=n_directions)) % n
    directions = Z[first] - Z[second]
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-12 * norms.max()]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    outlyingness = np.zeros(n)
    for projection in directions @ Z.T:
        center, scale = _unimcd(projection, h)
        if scale > 1e-12 * np.abs(projection).max():
            outlyingness = np.maximum(outlyingness, np.abs(projection - center) / scale)
    return outlyingness


def _orthogonal_cutoff(distances, h):
    """Cutoff ``(m + s z_0.975)^(3/2)`` for orthogonal distances (Wilson-Hilferty)."""
    center, scale = _unimcd(distances ** (2 / 3), h)
    return float((center + scale * _Z975) ** 1.5)


def _spatial_median(X, tol=1e-10, max_iter=500):
    """Point that minimizes the sum of Euclidean distances (Weiszfeld's algorithm)."""
    median = np.median(X, axis=0)
    spread = np.abs(X - median).max()
    for _ in range(max_iter if spread > 0 else 0):  # pragma: no branch (converges before the cap)
        distances = np.linalg.norm(X - median, axis=1)
        weights = 1 / np.maximum(distances, tol * spread)
        updated = weights @ X / weights.sum()
        converged = np.abs(updated - median).max() <= tol * spread
        median = updated
        if converged:
            break
    return median


def _grid_direction(Y, scale_fn, n_grid, n_cycles):
    """Unit direction with a locally maximal scale of ``Y @ direction`` (grid algorithm)."""
    scales = np.array([scale_fn(column) for column in Y.T])
    order = np.argsort(-scales, kind="stable")
    direction = np.zeros(Y.shape[1])
    direction[order[0]] = 1.0
    steps = np.arange(-(n_grid // 2), n_grid // 2 + 1) / max(n_grid // 2, 1)
    for cycle in range(n_cycles if Y.shape[1] > 1 else 0):
        angles = steps * np.pi / 2 ** (cycle + 1)
        cos, sin = np.cos(angles), np.sin(angles)
        for j in order:
            if 1 - abs(direction[j]) < 1e-12:  # the direction is this axis: no plane
                continue
            # Unit directions cos * direction + sin * e_j, before normalization.
            norms = np.sqrt(1 + 2 * cos * sin * direction[j])
            candidates = (np.outer(Y @ direction, cos) + np.outer(Y[:, j], sin)) / norms
            best = np.argmax([scale_fn(candidate) for candidate in candidates.T])
            direction = cos[best] * direction
            direction[j] += sin[best]
            direction /= norms[best]
    return direction


class _RobustDiscriminant(ClassifierMixin, BaseEstimator):
    """Base class: validation, priors and prediction from the discriminant scores."""

    def __init__(self, *, alpha=0.5, priors=None, random_state=None):
        self.alpha = alpha
        self.priors = priors
        self.random_state = random_state

    def _validate(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_samples=2)
        check_classification_targets(y)
        self.classes_, labels, counts = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError("y needs at least two classes")
        if counts.min() <= X.shape[1]:
            raise ValueError("every class needs more observations than variables")
        self.priors_ = counts / len(y)
        if self.priors is not None:
            self.priors_ = np.asarray(self.priors, dtype=float)
            if self.priors_.shape != counts.shape or np.any(self.priors_ < 0):
                raise ValueError("priors must be one nonnegative number per class")
            if not np.isclose(self.priors_.sum(), 1):
                raise ValueError("priors must sum to 1")
        return X, labels

    def _group_mcd(self, X, rng):
        return MCD(alpha=self.alpha, random_state=rng).fit(X)

    def _scores(self, X):
        """Discriminant score of each class in columns: log density plus log prior."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        covariances = np.broadcast_to(self._covariances(), (len(self.classes_), *X.shape[1:] * 2))
        scores = [
            -0.5 * (_mahalanobis_sq(X, mean, covariance) + _logdet(covariance))
            for mean, covariance in zip(self.means_, covariances, strict=True)
        ]
        with np.errstate(divide="ignore"):  # a zero prior excludes its class
            return np.column_stack(scores) + np.log(self.priors_)

    def decision_function(self, X):
        """Discriminant scores; with two classes, the score of the second minus the first."""
        scores = self._scores(X)
        return scores[:, 1] - scores[:, 0] if scores.shape[1] == 2 else scores

    def predict(self, X):
        """Class with the largest discriminant score."""
        best = self._scores(X).argmax(axis=1)
        return self.classes_[best]

    def predict_proba(self, X):
        """Posterior probabilities of the classes under the normal model."""
        scores = self._scores(X)
        likelihood = np.exp(scores - scores.max(axis=1, keepdims=True))
        return likelihood / likelihood.sum(axis=1, keepdims=True)


class RobustLDA(_RobustDiscriminant):
    """Robust linear discriminant analysis with the MCD (Hubert & Van Driessen 2004).

    The classical rule assigns ``x`` to the class with the largest
    ``mu_k' S^-1 x - mu_k' S^-1 mu_k / 2 + log(prior_k)``; a few outliers can
    ruin its sample means and pooled covariance ``S``. Here each class is
    centered at its reweighted :class:`~robustmodels.covariance.MCD` location,
    and the reweighted MCD of the pooled centered observations gives the common
    covariance and a correction of the centers (the pooled-observations
    estimator of He & Fung 2000). Every class needs more observations than
    variables.

    Parameters
    ----------
    alpha : float in [0.5, 1], default 0.5
        Coverage of the MCD estimators.
    priors : array-like of shape (n_classes,), default None
        Prior probabilities; the class proportions by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the MCD estimators.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    priors_ : ndarray of shape (n_classes,)
        Prior probabilities.
    means_ : ndarray of shape (n_classes, n_features)
        Robust class centers.
    covariance_ : ndarray of shape (n_features, n_features)
        Robust common covariance matrix.
    support_ : ndarray of bool, shape (n_samples,)
        Observations used by the reweighted pooled MCD; False marks outliers.

    References
    ----------
    Hubert, M. and Van Driessen, K. (2004). Fast and robust discriminant
    analysis. *Computational Statistics & Data Analysis*, 45(2), 301-320.

    He, X. and Fung, W. K. (2000). High breakdown estimation for multiple
    populations with applications to discriminant analysis. *Journal of
    Multivariate Analysis*, 72(2), 151-162.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.multivariate import RobustLDA
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(size=(50, 2)), rng.normal(size=(50, 2)) + 3])
    >>> y = np.repeat([0, 1], 50)
    >>> X[:5] = [15, 15]  # five outliers labeled as class 0
    >>> lda = RobustLDA(random_state=0).fit(X, y)
    >>> lda.means_.round(1).tolist()
    [[-0.1, 0.2], [3.1, 3.0]]
    >>> lda.predict([[0, 0], [3, 3]]).tolist()
    [0, 1]
    """

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and labels ``y``. Returns ``self``."""
        X, labels = self._validate(X, y)
        check_range("alpha", self.alpha, 0.5, 1)
        self.means_, pooled = _pooled_mcd(
            X, labels, lambda Z: self._group_mcd(Z, as_generator(self.random_state))
        )
        self.covariance_, self.support_ = pooled.covariance_, pooled.support_
        if _logdet(self.covariance_) == -np.inf:
            raise ValueError("the pooled covariance is singular")
        return self

    def _covariances(self):
        return self.covariance_


class RobustQDA(_RobustDiscriminant):
    """Robust quadratic discriminant analysis with the MCD (Hubert & Van Driessen 2004).

    Assigns ``x`` to the class with the largest
    ``-log det(S_k) / 2 - d_k(x)^2 / 2 + log(prior_k)``, where the center and
    covariance ``S_k`` of each class are its reweighted
    :class:`~robustmodels.covariance.MCD` estimates and ``d_k`` is the robust
    distance to class ``k``. Every class needs more observations than variables.

    Parameters
    ----------
    alpha : float in [0.5, 1], default 0.5
        Coverage of the MCD estimators.
    priors : array-like of shape (n_classes,), default None
        Prior probabilities; the class proportions by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the MCD estimators.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    priors_ : ndarray of shape (n_classes,)
        Prior probabilities.
    means_ : ndarray of shape (n_classes, n_features)
        Robust class centers.
    covariances_ : ndarray of shape (n_classes, n_features, n_features)
        Robust covariance matrix of each class.

    References
    ----------
    Hubert, M. and Van Driessen, K. (2004). Fast and robust discriminant
    analysis. *Computational Statistics & Data Analysis*, 45(2), 301-320.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.multivariate import RobustQDA
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(size=(50, 2)), 3 * rng.normal(size=(50, 2)) + 4])
    >>> y = np.repeat(["a", "b"], 50)
    >>> qda = RobustQDA(random_state=0).fit(X, y)
    >>> qda.predict([[0, 0], [8, 8]]).tolist()
    ['a', 'b']
    """

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and labels ``y``. Returns ``self``."""
        X, labels = self._validate(X, y)
        check_range("alpha", self.alpha, 0.5, 1)
        rng = as_generator(self.random_state)
        fits = [self._group_mcd(X[labels == k], rng) for k in range(len(self.classes_))]
        self.means_ = np.array([fit.location_ for fit in fits])
        self.covariances_ = np.array([fit.covariance_ for fit in fits])
        if any(_logdet(covariance) == -np.inf for covariance in self.covariances_):
            raise ValueError("the covariance of a class is singular")
        return self

    def _covariances(self):
        return self.covariances_


def _pooled_mcd(X, labels, fit_mcd):
    """Class centers and the MCD fit of the observations centered by class."""
    centers = np.array([fit_mcd(X[labels == k]).location_ for k in range(labels.max() + 1)])
    pooled = fit_mcd(X - centers[labels])
    return centers + pooled.location_, pooled


class HotellingResult(NamedTuple):
    """Result of :func:`hotelling_t2`."""

    statistic: float
    f_statistic: float
    df: tuple
    pvalue: float
    location: np.ndarray


def hotelling_t2(X, mu=None, *, random_state=None):
    """Robust one-sample Hotelling test (Willems, Pison, Rousseeuw & Van Aelst 2002).

    Tests that the center of ``X`` equals ``mu`` with
    ``T^2 = n (t - mu)' C^-1 (t - mu)``, where ``t`` and ``C`` are the
    reweighted :class:`~robustmodels.covariance.MCD` location and covariance
    with coverage 0.75. Under the normal model ``T^2 / d`` is approximately
    ``F(p, q)``; ``d`` and ``q`` match the mean and variance of ``T^2``, which
    the authors fitted to simulations for each ``n`` and ``p`` and which tend
    to those of the asymptotic chi-square distribution. For paired samples,
    test the differences.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    mu : array-like of shape (n_features,), default None
        Hypothesized center; the origin by default.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the MCD.

    Returns
    -------
    HotellingResult
        Named tuple with ``statistic`` (``T^2``), ``f_statistic`` (``T^2 / d``),
        ``df`` (``(p, q)``), ``pvalue`` and ``location`` (the MCD center).

    References
    ----------
    Willems, G., Pison, G., Rousseeuw, P. J. and Van Aelst, S. (2002). A robust
    Hotelling test. *Metrika*, 55, 125-138.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.multivariate import hotelling_t2
    >>> X = np.random.default_rng(0).normal(size=(60, 3))
    >>> X[:6] += 8  # outliers that would shift the sample mean
    >>> hotelling_t2(X, random_state=0).pvalue > 0.05
    True
    >>> hotelling_t2(X, mu=[1, 1, 1], random_state=0).pvalue < 0.001
    True
    """
    mcd = MCD(alpha=0.75, random_state=random_state).fit(X)
    n, p = len(mcd.dist_), len(mcd.location_)
    mu = np.zeros(p) if mu is None else np.asarray(mu, dtype=float)
    if mu.shape != (p,) or not np.isfinite(mu).all():
        raise ValueError(f"mu must contain {p} finite values")
    if _logdet(mcd.covariance_) == -np.inf:
        raise ValueError("the MCD covariance is singular")
    statistic = float(n * _mahalanobis_sq(mu[None, :], mcd.location_, mcd.covariance_)[0])
    d, q = _t2_approximation(p, n)
    pvalue = float(stats.f.sf(statistic / d, p, q))
    return HotellingResult(statistic, statistic / d, (p, q), pvalue, mcd.location_)


# Willems et al. (2002): mean and variance of T^2 minus their limits follow
# exp(a) * n^b. Rows give (a, b) for p = 1 and p = 2; for p > 2 the curve passes
# through exp(c1) / p^c2 at n = 7 p^2 and n = 10 p^2.
_T2_MEAN = {1: (2.741737, -1.403526), 2: (3.383723, -1.081893)}
_T2_MEAN_POINTS = ((1.050505, 1.360808), (0.7280273, 1.4685161))
_T2_VARIANCE = {1: (6.202284, -1.731468), 2: (8.518773, -1.851881)}
_T2_VARIANCE_POINTS = ((4.829586, 2.868279), (4.261745, 3.023672))


def _t2_excess(p, n, small, points):
    if p in small:
        a, b = small[p]
    else:
        (c1, c2), (c3, c4) = points
        y1, y2 = c1 - c2 * np.log(p), c3 - c4 * np.log(p)
        b = (y2 - y1) / np.log(10 / 7)
        a = y1 - b * np.log(7 * p * p)
    return np.exp(a) * n**b


def _t2_approximation(p, n):
    """Factor ``d`` and degrees of freedom ``q`` with ``T^2 / d ~ F(p, q)``."""
    kappa = _reweighted_mcd_variance(p)
    mean = kappa * p + _t2_excess(p, n, _T2_MEAN, _T2_MEAN_POINTS)
    variance = 2 * p * kappa**2 + _t2_excess(p, n, _T2_VARIANCE, _T2_VARIANCE_POINTS)
    q = (p + 2) / (variance / mean**2 * p / 2 - 1) + 4
    if not 0 <= q <= n:
        q = float(n)
    return float(mean * (q - 2) / q), float(q)


def _reweighted_mcd_variance(p, alpha=0.75, delta=0.025):
    """Asymptotic variance factor of the reweighted MCD location at the normal model.

    Croux & Haesbroeck (1999) for the raw MCD with coverage ``alpha``, and
    Lopuhaä (1999) for the reweighting step with cutoff quantile ``1 - delta``.
    """
    raw = chi2.cdf(chi2.ppf(alpha, p), p + 2)  # 1 / variance factor of the raw location
    kept = chi2.cdf(chi2.ppf(1 - delta, p), p + 2)
    shrink = 1 - kept / (1 - delta)
    return float(shrink**2 / raw + 2 * shrink / (1 - delta) + kept / (1 - delta) ** 2)


class ManovaResult(NamedTuple):
    """Result of :func:`manova`."""

    statistic: float
    chi2_statistic: float
    df: float
    pvalue: float
    means: np.ndarray


def manova(X, groups, *, alpha=0.5, n_simulations=500, random_state=None):
    """Robust one-way MANOVA with Wilks' lambda (Todorov & Filzmoser 2010).

    Tests that all groups have the same mean vector, assuming a common
    covariance matrix. Observations get weight 1 when their squared robust
    distance to their own group center is below the 0.975 chi-square quantile
    and 0 otherwise; centers and common covariance are the reweighted pooled
    :class:`~robustmodels.covariance.MCD` estimates of :class:`RobustLDA`,
    computed with the deterministic algorithm. Wilks' lambda is ``det(W) / det(T)``, with ``W``
    and ``T`` the within-group and total sums of squares and products of the
    observations with weight 1.

    Under the normal model the null distribution depends only on the group
    sizes and the dimension, so it is simulated with standard normal samples
    and the same estimator: ``log(lambda) / d`` is treated as chi-square with
    ``q`` degrees of freedom, where ``d`` and ``q`` match the simulated mean
    and variance of ``log(lambda)``. With two groups this is a robust
    two-sample Hotelling test. The simulation takes most of the time, which
    is why the deterministic MCD is used; R's ``rrcov::Wilks.test`` searches
    with FAST-MCD and can find other subsets, hence another lambda.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    groups : array-like of shape (n_samples,)
        Group label of each observation.
    alpha : float in [0.5, 1], default 0.5
        Coverage of the MCD estimators.
    n_simulations : int, default 500
        Normal samples used to approximate the null distribution. Each needs
        one MCD per group and one for the pooled data.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the simulation.

    Returns
    -------
    ManovaResult
        Named tuple with ``statistic`` (Wilks' lambda), ``chi2_statistic``
        (``log(lambda) / d``), ``df`` (``q``), ``pvalue`` and ``means`` (the
        weighted group means, in the sorted order of the labels).

    References
    ----------
    Todorov, V. and Filzmoser, P. (2010). Robust statistic for the one-way
    MANOVA. *Computational Statistics & Data Analysis*, 54(1), 37-48.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.multivariate import manova
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(90, 2))
    >>> groups = np.repeat([0, 1, 2], 30)
    >>> X[:4] += 9  # outliers in the first group
    >>> manova(X, groups, n_simulations=100, random_state=0).pvalue > 0.05
    True
    >>> X[groups == 2] += 1.5  # a real shift
    >>> manova(X, groups, n_simulations=100, random_state=0).pvalue < 0.001
    True
    """
    rule = RobustLDA(alpha=alpha)
    X, labels = rule._validate(X, groups)
    check_range("alpha", alpha, 0.5, 1)
    check_count("n_simulations", n_simulations)
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least 2")
    rng = as_generator(random_state)
    fit_mcd = MCD(alpha=alpha, algorithm="deterministic").fit
    statistic, means = _wilks(X, labels, fit_mcd)
    null = [_wilks(rng.normal(size=X.shape), labels, fit_mcd)[0] for _ in range(n_simulations)]
    logs = np.log(null)
    q = 2 * logs.mean() ** 2 / logs.var(ddof=1)
    chi2_statistic = float(np.log(statistic) * q / logs.mean())
    pvalue = float(chi2.sf(chi2_statistic, q))
    return ManovaResult(statistic, chi2_statistic, float(q), pvalue, means)


def _wilks(X, labels, fit_mcd):
    """Wilks' lambda and group means of the observations with weight 1."""
    kept = _pooled_mcd(X, labels, fit_mcd)[1].dist_ < chi2.ppf(0.975, X.shape[1])
    X, labels = X[kept], labels[kept]
    groups = [X[labels == k] for k in range(labels.max() + 1)]
    if min(len(group) for group in groups) == 0:
        raise ValueError("a group has no observations left after removing outliers")
    means = np.array([group.mean(axis=0) for group in groups])
    residuals = X - means[labels]
    centered = X - X.mean(axis=0)
    within, total = residuals.T @ residuals, centered.T @ centered
    statistic = np.exp(np.linalg.slogdet(within)[1] - np.linalg.slogdet(total)[1])
    return float(statistic), means
