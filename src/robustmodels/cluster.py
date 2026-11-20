"""Robust clustering by trimming: TCLUST and trimmed k-means.

:class:`TClust` follows scikit-learn's clustering API. ``fit(X)`` sets
``labels_`` (``-1`` for trimmed observations), the cluster ``means_``,
``covariances_`` and ``weights_``; ``predict(X)`` assigns new observations and
trims those less plausible than every retained training observation.
"""

import math

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_generator, check_count, check_range

__all__ = ["TClust"]

_ZERO = 1e-16  # eigenvalues up to this are zero, as tclust's zero_tol


class TClust(ClusterMixin, BaseEstimator):
    """Robust clustering by trimming and eigenvalue restrictions (García-Escudero et al. 2008).

    Finds ``n_clusters`` normal clusters that maximize the classification
    likelihood ``sum log(w_j phi(x_i; mu_j, Sigma_j))`` of the observations
    they keep, after discarding a fraction ``alpha`` of the sample: the
    observations whose best cluster explains them worst. The ratio between the
    largest and smallest eigenvalue of all the covariance matrices is at most
    ``restr_factor``, which keeps the problem well posed (otherwise a cluster
    could collapse on a few points) and controls how different the cluster
    shapes may be; ``restr_factor=1`` with ``equal_weights=True`` gives trimmed
    k-means (Cuesta-Albertos, Gordaliza & Matrán 1997), with spherical clusters
    of equal size. The restriction is applied by the optimal truncation of
    Fritz, García-Escudero & Mayo-Iscar (2013).

    The search follows R's tclust: ``n_init`` random starts (each cluster from
    ``p + 1`` random observations) improved by 3 concentration steps, then the
    best 5 by up to ``max_iter`` steps each; a step assigns and trims the
    observations and re-estimates the clusters. It finds a local maximum; more
    starts make the global one more likely.

    Parameters
    ----------
    n_clusters : int, default 2
    alpha : float in [0, 1), default 0.05
        Fraction of observations trimmed.
    restr_factor : float >= 1, default 12.0
        Largest allowed ratio between eigenvalues of the covariance matrices.
    equal_weights : bool, default False
        Clusters of equal size (weights ``1 / n_clusters``).
    n_init : int, default 500
        Random starts.
    max_iter : int, default 20
        Concentration steps for each of the best starts.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster of each observation, ``-1`` for trimmed ones.
    means_ : ndarray of shape (n_clusters, n_features)
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Restricted covariance matrices.
    weights_ : ndarray of shape (n_clusters,)
        Fitted cluster proportions, or equal weights. At the iteration cap,
        these are the weights used for the last assignment.
    objective_ : float
        Trimmed classification log-likelihood of the solution.
    threshold_ : float
        Smallest ``log(w_j phi(x; mu_j, Sigma_j))`` of a retained observation in
        its cluster; ``predict`` trims observations below it (up to rounding).
    n_iter_ : int
        Concentration steps of the selected solution after its start.
    converged_ : bool
        Whether the selected solution reached a stable assignment.

    References
    ----------
    García-Escudero, L. A., Gordaliza, A., Matrán, C. and Mayo-Iscar, A.
    (2008). A general trimming approach to robust cluster analysis. *The Annals
    of Statistics*, 36(3), 1324-1345.

    Fritz, H., García-Escudero, L. A. and Mayo-Iscar, A. (2012). tclust: An R
    package for a trimming approach to cluster analysis. *Journal of
    Statistical Software*, 47(12), 1-26.

    Fritz, H., García-Escudero, L. A. and Mayo-Iscar, A. (2013). A fast
    algorithm for robust constrained clustering. *Computational Statistics &
    Data Analysis*, 61, 124-136.

    Cuesta-Albertos, J. A., Gordaliza, A. and Matrán, C. (1997). Trimmed
    k-means: an attempt to robustify quantizers. *The Annals of Statistics*,
    25(2), 553-576.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.cluster import TClust
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(6, 1, (60, 2))])
    >>> X[:6] = rng.uniform(-20, 20, (6, 2))  # scattered noise
    >>> fit = TClust(n_clusters=2, alpha=0.05, random_state=0).fit(X)
    >>> np.flatnonzero(fit.labels_ == -1).tolist()  # point 0 fell near a cluster
    [1, 2, 3, 4, 5, 109]
    >>> (fit.means_.round(1) + 0.0).tolist()
    [[5.7, 6.0], [0.0, 0.2]]
    """

    def __init__(
        self,
        n_clusters=2,
        *,
        alpha=0.05,
        restr_factor=12.0,
        equal_weights=False,
        n_init=500,
        max_iter=20,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.restr_factor = restr_factor
        self.equal_weights = equal_weights
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        check_count("n_clusters", self.n_clusters)
        check_count("n_init", self.n_init)
        check_count("max_iter", self.max_iter)
        check_range("alpha", self.alpha, 0, 1, high_open=True)
        check_range("restr_factor", self.restr_factor, 1, np.inf, high_open=True)
        kept = math.floor(len(X) * (1 - self.alpha))
        if kept < self.n_clusters:
            raise ValueError("alpha trims too many observations for n_clusters clusters")
        origin = X[0]
        X = X - origin
        scale = np.max(np.abs(X)) or 1.0
        X = X / scale
        rng = as_generator(self.random_state)
        starts = [
            self._concentrate(X, self._random_start(X, kept, rng), kept, 3)
            for _ in range(self.n_init)
        ]
        starts.sort(key=lambda solution: -solution["objective"])
        best = None
        for start in starts[:5]:
            parameters = self._estimate(X, start["labels"], kept)
            solution = self._concentrate(X, parameters, kept, self.max_iter)
            if best is None or solution["objective"] > best["objective"]:
                best = solution
        order = np.argsort(-best["sizes"], kind="stable")  # largest cluster first
        relabel = np.full(self.n_clusters + 1, -1)
        relabel[order] = np.arange(self.n_clusters)
        self.labels_ = relabel[best["labels"]]
        self.means_ = best["means"][order] * scale + origin
        self.covariances_ = best["covariances"][order] * scale * scale
        self.weights_ = best["weights"][order]
        adjustment = X.shape[1] * np.log(scale)
        self.objective_ = best["objective"] - kept * adjustment
        self.threshold_ = best["threshold"] - adjustment
        self.n_iter_ = best["n_iter"]
        self.converged_ = best["converged"]
        return self

    def predict(self, X):
        """Cluster of each row of ``X``; ``-1`` below the training ``threshold_``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        scores = _discriminant(X, self.means_, self.covariances_, self.weights_)
        cutoff = self.threshold_ - 1e-9 * max(1.0, abs(self.threshold_))
        return np.where(scores.max(axis=1) >= cutoff, scores.argmax(axis=1), -1)

    def _random_start(self, X, kept, rng):
        """Each cluster from ``p + 1`` observations drawn with replacement, as tclust."""
        n, p = X.shape
        k = self.n_clusters
        points = X[rng.integers(0, n, size=(k, p + 1))]
        means = points[:, 0].copy()
        centred = points - points.mean(axis=1, keepdims=True)
        covariances = np.einsum("kij,kil->kjl", centred, centred) / (p + 1)
        weights = np.full(k, 1 / k) if self.equal_weights else rng.random(k)
        weights = weights / weights.sum()
        sizes = np.full(k, kept / k) if self.equal_weights else np.round(n * weights)
        return {"means": means, "covariances": covariances, "weights": weights, "sizes": sizes}

    def _estimate(self, X, labels, kept):
        """Means, covariances (divisor ``n_j``) and weights of the clusters ``labels``."""
        k, p = self.n_clusters, X.shape[1]
        means, covariances = np.zeros((k, p)), np.tile(np.eye(p), (k, 1, 1))
        sizes = np.bincount(labels[labels >= 0], minlength=k).astype(float)
        for j in np.flatnonzero(sizes):
            members = X[labels == j]
            means[j] = members.mean(axis=0)
            centred = members - means[j]
            covariances[j] = centred.T @ centred / sizes[j]
        weights = np.full(k, 1 / k) if self.equal_weights else sizes / kept
        return {"means": means, "covariances": covariances, "weights": weights, "sizes": sizes}

    def _concentrate(self, X, parameters, kept, steps):
        """Concentration steps: restrict, assign and trim, then re-estimate."""
        labels = None
        last_scored = parameters
        converged = False
        for step in range(steps):
            covariances = _restrict(
                parameters["covariances"], parameters["sizes"], self.restr_factor
            )
            if covariances is None:  # every eigenvalue is zero
                if step:
                    parameters = last_scored
                    break
                covariances = np.tile(np.eye(X.shape[1]), (self.n_clusters, 1, 1))
            parameters["covariances"] = covariances
            scores = _discriminant(X, parameters["means"], covariances, parameters["weights"])
            best = scores.max(axis=1)
            new = np.full(len(X), -1)
            retained = np.argsort(-best, kind="stable")[:kept]
            new[retained] = scores[retained].argmax(axis=1)
            converged = labels is not None and np.array_equal(new, labels)
            labels = new
            if converged:
                break
            if step < steps - 1:  # no re-estimation after the last assignment, as tclust
                last_scored = parameters
                parameters = self._estimate(X, labels, kept)
        retained = labels >= 0
        fitted = scores[retained, labels[retained]]
        sizes = np.bincount(labels[retained], minlength=self.n_clusters).astype(float)
        return {
            **parameters,
            "sizes": sizes,
            "labels": labels,
            "objective": float(fitted.sum()),
            "threshold": float(fitted.min()),
            "n_iter": step + 1,
            "converged": converged,
        }


def _discriminant(X, means, covariances, weights):
    """``log(w_j) + log phi(x_i; mu_j, Sigma_j)`` for each observation and cluster."""
    p = X.shape[1]
    out = np.empty((len(X), len(means)))
    for j, (mean, covariance, weight) in enumerate(zip(means, covariances, weights, strict=True)):
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, values.max() * np.finfo(float).eps)
        z = (X - mean) @ vectors / np.sqrt(values)
        logdet = np.log(values).sum()
        with np.errstate(divide="ignore"):
            out[:, j] = np.log(weight) - 0.5 * (p * np.log(2 * np.pi) + logdet + (z**2).sum(1))
    return out


def _restrict(covariances, sizes, factor):
    """Covariances whose eigenvalues all lie in ``[m, factor * m]``, or None if all are zero.

    The eigenvalues ``d`` are truncated to ``min(max(d, m), factor m)`` with the
    ``m`` that minimizes ``sum_j w_j sum_l (log e_jl + d_jl / e_jl)`` for the
    truncated values ``e`` (the likelihood of the truncated covariances), as
    Fritz, García-Escudero & Mayo-Iscar (2013).
    """
    values, vectors = np.linalg.eigh(covariances)
    values = np.maximum(values, 0)
    restricted = _restrict_eigenvalues(values, sizes, factor)
    if restricted is None:
        return None
    return np.einsum("kij,kj,klj->kil", vectors, restricted, vectors)


def _restrict_eigenvalues(d, sizes, factor):
    """Optimal truncation of the eigenvalues ``d`` (clusters in rows)."""
    nonempty = sizes > 0
    if not nonempty.any() or d[nonempty].max() <= _ZERO:
        return None
    if d[nonempty].max() <= factor * d[nonempty].min():
        d = d.copy()
        d[~nonempty] = d[nonempty].mean()
        return d
    w = sizes / sizes.sum()
    knots = np.sort(np.r_[d.ravel(), d.ravel() / factor])
    edges = np.r_[0, knots, 2 * knots[-1]]
    midpoints = (edges[:-1] + edges[1:])[:, None, None] / 2
    low, high = d < midpoints, d > factor * midpoints
    count = (w * (low.sum(2) + high.sum(2))).sum(1)
    total = (w * ((d * low).sum(2) + (d * high).sum(2) / factor)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidates = total / count
    candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
    truncated = np.clip(d, candidates[:, None, None], factor * candidates[:, None, None])
    loss = (w[:, None] * (np.log(truncated) + d / truncated)).sum((1, 2))
    m = candidates[np.argmin(loss)]
    return np.clip(d, m, factor * m)
