"""Multivariate kernel density estimation with a full bandwidth matrix.

:class:`KernelDensity` estimates a density, its gradient and its Hessian with a
normal kernel whose bandwidth is a full matrix ``H``, chosen by
:func:`plugin_bandwidth` or :func:`scv_bandwidth` for the density or for one of
its derivatives. ``hdr_level`` gives the density levels that bound highest
density regions, and :func:`kde_test` tests whether two samples have the same
density.

The selectors need integrals of products of normal density derivatives. For a
normal kernel these are sums over pairs of observations of a normal density
times moments of quadratic forms of a normal vector (Chacón & Duong 2015),
which are computed exactly from their cumulants.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy import linalg, optimize, special, stats
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import check_option

__all__ = ["KDETestResult", "KernelDensity", "kde_test", "plugin_bandwidth", "scv_bandwidth"]

_DERIVATIVES = (0, 1, 2)
_BLOCK = 2**18  # pairs of observations per block of a double sum


class KernelDensity(BaseEstimator):
    """Kernel density estimator with a normal kernel and a full bandwidth matrix.

    The estimate at ``x`` is ``f(x) = mean_i phi_H(x - X_i)``, with ``phi_H``
    the normal density with covariance ``H``. A full ``H`` smooths along the
    directions of the data, which a scalar or diagonal bandwidth cannot
    (Wand & Jones 1993). By default ``H`` minimizes a plug-in estimate of the
    mean integrated squared error of the density or of its derivative of
    order ``deriv_order``: derivatives need more smoothing than the density.

    Parameters
    ----------
    bandwidth : {"plugin", "scv"}, float or array-like, default "plugin"
        ``"plugin"`` and ``"scv"`` select ``H`` with :func:`plugin_bandwidth`
        or :func:`scv_bandwidth`; a number ``h`` gives ``H = h^2 I``, and a
        matrix of shape (n_features, n_features) is used as ``H``.
    deriv_order : {0, 1, 2}, default 0
        Derivative the selected bandwidth is optimal for: the density (0), its
        gradient (1) or its Hessian (2).

    Attributes
    ----------
    bandwidth_ : ndarray of shape (n_features, n_features)
        Bandwidth matrix ``H``.
    X_ : ndarray of shape (n_samples, n_features)
        Training data.

    References
    ----------
    Wand, M. P. and Jones, M. C. (1993). Comparison of smoothing
    parameterizations in bivariate kernel density estimation. *Journal of the
    American Statistical Association*, 88(422), 520-528.

    Chacón, J. E. and Duong, T. (2018). *Multivariate Kernel Smoothing and Its
    Applications*. Chapman and Hall/CRC.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.density import KernelDensity
    >>> rng = np.random.default_rng(0)
    >>> X = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=300)
    >>> kde = KernelDensity().fit(X)
    >>> kde.bandwidth_.round(3).tolist()  # oriented along the correlation
    [[0.087, 0.064], [0.064, 0.069]]
    >>> np.exp(kde.score_samples([[0, 0], [1, -1]])).round(3).tolist()
    [0.182, 0.004]
    """

    def __init__(self, bandwidth="plugin", *, deriv_order=0):
        self.bandwidth = bandwidth
        self.deriv_order = deriv_order

    def fit(self, X, y=None):
        """Fit to ``X`` of shape (n_samples, n_features); ``y`` is ignored. Returns ``self``."""
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        check_option("deriv_order", self.deriv_order, _DERIVATIVES)
        d = X.shape[1]
        if isinstance(self.bandwidth, str):
            check_option("bandwidth", self.bandwidth, ("plugin", "scv"))
            select = plugin_bandwidth if self.bandwidth == "plugin" else scv_bandwidth
            H = select(X, deriv_order=self.deriv_order)
        elif np.ndim(self.bandwidth) == 0:
            if not 0 < self.bandwidth < np.inf:
                raise ValueError(f"bandwidth must be positive, got {self.bandwidth!r}")
            H = float(self.bandwidth) ** 2 * np.eye(d)
        else:
            H = np.array(self.bandwidth, dtype=float)
            if H.shape != (d, d) or not np.allclose(H, H.T) or np.linalg.eigvalsh(H)[0] <= 0:
                raise ValueError(f"bandwidth must be a symmetric positive definite {d}x{d} matrix")
        self.bandwidth_ = H
        self.X_ = X
        return self

    def score_samples(self, X):
        """Log density at each row of ``X``."""
        out = [
            special.logsumexp(-np.einsum("ijk,ijk->ij", W, W) / 2, axis=1) for W in self._whiten(X)
        ]
        d = self.n_features_in_
        logdet = np.linalg.slogdet(self.bandwidth_)[1]
        return np.concatenate(out) - np.log(len(self.X_)) - (d * np.log(2 * np.pi) + logdet) / 2

    def gradient(self, X):
        """Gradient of the density estimate at each row of ``X``, shape (n_queries, n_features)."""
        root = self._root()
        out = [-np.einsum("ij,ijk->ik", self._kernel(W), W) for W in self._whiten(X)]
        return linalg.solve_triangular(root, np.concatenate(out).T, lower=True, trans=1).T

    def hessian(self, X):
        """Hessian of the density estimate at each row of ``X``.

        Returns an array of shape (n_queries, n_features, n_features).
        """
        root, d = self._root(), self.n_features_in_
        out = []
        for W in self._whiten(X):
            k = self._kernel(W)
            out.append(np.einsum("ij,ijk,ijl->ikl", k, W, W) - k.sum(1)[:, None, None] * np.eye(d))
        inverse = linalg.solve_triangular(root, np.eye(d), lower=True)  # L^-1, with H = L L'
        return inverse.T @ np.concatenate(out) @ inverse

    def hdr_level(self, prob):
        """Density level whose upper level set is the ``prob`` highest density region.

        The region ``{x : f(x) >= level}`` has probability ``prob``; the level is
        estimated by the ``1 - prob`` quantile of the density estimate at the
        training observations (Hyndman 1996), as R's ``ks::contourLevels``.
        ``prob`` may be a number or an array of numbers in (0, 1).

        Hyndman, R. J. (1996). Computing and graphing highest density regions.
        *The American Statistician*, 50(2), 120-126.
        """
        check_is_fitted(self)
        prob = np.asarray(prob, dtype=float)
        if not np.all((prob > 0) & (prob < 1)):
            raise ValueError(f"prob must lie in (0, 1), got {prob}")
        return np.quantile(np.exp(self.score_samples(self.X_)), 1 - prob)

    def _root(self):
        return np.linalg.cholesky(self.bandwidth_)

    def _whiten(self, X):
        """Blocks of ``L^-1 (x - X_i)`` for queries ``x`` (rows) and training rows ``X_i``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        root = self._root()
        train = linalg.solve_triangular(root, self.X_.T, lower=True).T
        query = linalg.solve_triangular(root, X.T, lower=True).T
        step = max(1, _BLOCK // len(train))
        for start in range(0, len(query), step):
            yield query[start : start + step, None, :] - train[None, :, :]

    def _kernel(self, W):
        """Normal kernel ``phi_H`` over ``n`` for whitened differences ``W``."""
        d = self.n_features_in_
        scale = len(self.X_) * np.sqrt((2 * np.pi) ** d * np.linalg.det(self.bandwidth_))
        return np.exp(-np.einsum("ijk,ijk->ij", W, W) / 2) / scale


def plugin_bandwidth(X, deriv_order=0):
    """Plug-in bandwidth matrix of a normal kernel (Chacón & Duong 2010, 2013).

    Minimizes over all symmetric positive definite ``H`` the asymptotic mean
    integrated squared error of the estimate of the derivative of order ``r``,

        ``n^-1 |H|^-1/2 2^-(d+r) pi^-d/2 nu_r(H^-1)
        + (-1)^r / 4 (vec I_{d^r} (x) vec H (x) vec H)' psi_{2r+4}``,

    where ``nu_r(A) = E[(Z'AZ)^r]`` for standard normal ``Z`` and the
    integrated density derivative ``psi_{2r+4}`` is estimated with a scalar
    pilot bandwidth on the sphered data, chosen in two stages from a normal
    reference (the ``dscalar`` pilot of Chacón & Duong 2013). The result is
    affine equivariant: transforming ``X`` by a matrix ``A`` gives ``A H A'``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample with a nonsingular covariance matrix.
    deriv_order : {0, 1, 2}, default 0
        Order of the derivative to estimate: density, gradient or Hessian.

    Returns
    -------
    ndarray of shape (n_features, n_features)

    Notes
    -----
    The double sums over pairs take ``O(n^2)`` time; R's ``ks::Hpi`` bins the
    data instead for large samples, and its default pilot for the density of
    two variables is the SAMSE pilot (pass ``pilot = "dscalar"`` to compare).

    References
    ----------
    Chacón, J. E. and Duong, T. (2010). Multivariate plug-in bandwidth
    selection with unconstrained pilot bandwidth matrices. *Test*, 19(2),
    375-398.

    Chacón, J. E. and Duong, T. (2013). Data-driven density derivative
    estimation, with applications to nonparametric clustering and bump
    hunting. *Electronic Journal of Statistics*, 7, 499-532.

    Chacón, J. E., Duong, T. and Wand, M. P. (2011). Asymptotics for general
    multivariate kernel density derivative estimators. *Statistica Sinica*,
    21(2), 807-840.
    """
    check_option("deriv_order", deriv_order, _DERIVATIVES)
    Z, root = _sphere(_check_sample(X))
    criterion = _plugin_criterion(Z, deriv_order)
    return root @ _minimize(criterion, _normal_scale(*Z.shape, deriv_order)) @ root


def scv_bandwidth(X, deriv_order=0):
    """Smoothed cross-validation bandwidth matrix of a normal kernel (Chacón & Duong 2011).

    Minimizes over all symmetric positive definite ``H`` the variance term of
    :func:`plugin_bandwidth` plus the smoothed estimate of the integrated
    squared bias,

        ``(-1)^r n^-2 sum_ij (vec I_{d^r})' D^{2r} L(X_i - X_j)``,
        ``L = phi_{2H+2G} - 2 phi_{H+2G} + phi_{2G}``,

    which replaces the asymptotic bias by its exact value for the data
    presmoothed with the pilot ``G = g^2 I`` on the sphered data (Duong &
    Hazelton 2005), with the ``dscalar`` pilot of Chacón & Duong (2013).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample with a nonsingular covariance matrix.
    deriv_order : {0, 1, 2}, default 0

    Returns
    -------
    ndarray of shape (n_features, n_features)

    Notes
    -----
    Each evaluation of the criterion sums over all pairs, so the selection
    takes ``O(n^2)`` time per step of the minimization.

    References
    ----------
    Duong, T. and Hazelton, M. L. (2005). Cross-validation bandwidth matrices
    for multivariate kernel density estimation. *Scandinavian Journal of
    Statistics*, 32(3), 485-506.

    Chacón, J. E. and Duong, T. (2011). Unconstrained pilot selectors for
    smoothed cross-validation. *Australian & New Zealand Journal of
    Statistics*, 53(3), 331-351.
    """
    check_option("deriv_order", deriv_order, _DERIVATIVES)
    Z, root = _sphere(_check_sample(X))
    criterion = _scv_criterion(Z, deriv_order)
    return root @ _minimize(criterion, _normal_scale(*Z.shape, deriv_order)) @ root


class KDETestResult(NamedTuple):
    """Result of :func:`kde_test`."""

    statistic: float
    z: float
    pvalue: float


def kde_test(X1, X2):
    """Two-sample test of equal densities by kernel estimates (Duong, Goud & Schauer 2012).

    The statistic estimates the integrated squared difference
    ``T = int (f1 - f2)^2 = psi_1 + psi_2 - psi_12 - psi_21``, where
    ``psi_1 = n1^-2 sum_ij phi_H1(X1_i - X1_j)`` estimates ``int f1^2`` and
    ``psi_12 = (n1 n2)^-1 sum_ij phi_H1(X1_i - X2_j)`` estimates
    ``int f1 f2``. Under the hypothesis ``f1 = f2``, ``T`` is asymptotically
    normal with mean ``(n1^-1 |H1|^-1/2 + n2^-1 |H2|^-1/2) phi(0)`` and the
    variance of the reference, estimated from the density gradient at the
    sample means; large values reject. ``Hj = h_j^2 S_j`` (``S_j`` the sample
    covariance) with ``h_j`` minimizing the asymptotic mean squared error of
    ``psi_j``, estimated with a normal reference pilot.

    Parameters
    ----------
    X1, X2 : array-like of shape (n1, n_features) and (n2, n_features)
        The samples; one-dimensional arrays are samples of one variable.

    Returns
    -------
    KDETestResult
        ``statistic`` (``T``), ``z`` (standardized ``T``) and the upper-tail
        normal ``pvalue``.

    Notes
    -----
    R's ``ks::kde.test`` minimizes the same mean squared error over all
    matrices ``H``, whose minimizers form a continuum; it returns the one its
    optimizer reaches. The bandwidth proportional to ``S`` is the unique
    minimizer of that form. Passing these bandwidths to ``kde.test`` gives the
    same statistic and p-value.

    References
    ----------
    Duong, T., Goud, B. and Schauer, K. (2012). Closed-form density-based
    framework for automatic detection of cellular morphology changes.
    *Proceedings of the National Academy of Sciences*, 109(22), 8382-8387.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.density import kde_test
    >>> rng = np.random.default_rng(0)
    >>> X1 = rng.normal(size=(150, 2))
    >>> X2 = rng.normal(size=(150, 2)) * [1, 2]  # more spread in the second variable
    >>> result = kde_test(X1, X2)
    >>> round(result.z, 2), round(result.pvalue, 4)
    (2.21, 0.0134)
    """
    X1, X2 = _check_sample(X1), _check_sample(X2)
    if X1.shape[1] != X2.shape[1]:
        raise ValueError(f"the samples have {X1.shape[1]} and {X2.shape[1]} variables")
    (n1, d), n2 = X1.shape, len(X2)
    H1, H2 = _functional_bandwidth(X1), _functional_bandwidth(X2)
    statistic = (
        _normal_sum(X1, X1, H1, 0)
        + _normal_sum(X2, X2, H2, 0)
        - _normal_sum(X1, X2, H1, 0)
        - _normal_sum(X2, X1, H2, 0)
    )
    mean = (1 / (n1 * np.sqrt(np.linalg.det(H1))) + 1 / (n2 * np.sqrt(np.linalg.det(H2)))) / (
        (2 * np.pi) ** (d / 2)
    )
    v1, v2 = _gradient_variance(X1), _gradient_variance(X2)
    variance = 3 * (n1 * v1 + n2 * v2) / (n1 + n2) * (1 / n1 + 1 / n2)
    z = (statistic - mean) / np.sqrt(variance)
    return KDETestResult(float(statistic), float(z), float(stats.norm.sf(z)))


def _check_sample(X):
    X = np.asarray(X, dtype=float)
    return check_array(X[:, None] if X.ndim == 1 else X, ensure_min_samples=2)


def _sphere(X):
    """``X S^-1/2`` and ``S^1/2``, with ``S`` the sample covariance and symmetric roots."""
    values, vectors = np.linalg.eigh(np.atleast_2d(np.cov(X, rowvar=False)))
    if values[0] <= 1e-12 * max(values[-1], 0):
        raise ValueError("the sample covariance of X is singular")
    return X @ (vectors / np.sqrt(values)) @ vectors.T, (vectors * np.sqrt(values)) @ vectors.T


def _normal_scale(n, d, r):
    """Normal scale bandwidth of the sphered data, R's ``ks::Hns``: the start of the search."""
    return (4 / (n * (d + 2 * r + 2))) ** (2 / (d + 2 * r + 4)) * np.eye(d)


def _differences(X, Y):
    """Blocks of the differences ``X_i - Y_j`` over all pairs."""
    step = max(1, _BLOCK // len(Y))
    for start in range(0, len(X), step):
        yield (X[start : start + step, None, :] - Y[None, :, :]).reshape(-1, X.shape[1])


def _moments(r, traces, quads):
    """``E[(u'u)^k]``, ``k = 0, ..., r``, for ``u ~ N(z, C)`` given ``tr(C^m)`` and ``z'C^(m-1)z``.

    ``traces[m]`` and ``quads[m]`` hold these for ``m = 1, ..., r``. The
    cumulants of ``u'u`` are ``2^(m-1) (m-1)! (tr(C^m) + m z'C^(m-1)z)``; the
    moments follow by the recursion between moments and cumulants. The
    formulas are polynomial in ``C``, so they also hold for the negative
    definite ``C`` of the Hermite polynomials of the normal density.
    """
    kappa = [0.0] + [
        2 ** (m - 1) * math.factorial(m - 1) * (traces[m] + m * quads[m]) for m in range(1, r + 1)
    ]
    mu = [1.0]
    for k in range(1, r + 1):
        mu.append(sum(math.comb(k - 1, m - 1) * kappa[m] * mu[k - m] for m in range(1, k + 1)))
    return mu


def _normal_sum(X, Y, Sigma, r):
    """``(n_X n_Y)^-1 sum_ij (vec I_{d^r})' D^{2r} phi_Sigma(X_i - Y_j)``, R's ``ks:::Qr``.

    The derivative of order ``2r`` of ``phi_Sigma`` at ``x`` is ``phi_Sigma(x)``
    times the expectation of ``u^{(x) 2r}`` with ``u ~ N(Sigma^-1 x,
    -Sigma^-1)`` (Holmquist 1996), and its contraction by ``vec I`` is
    ``E[(u'u)^r]``.
    """
    values, vectors = np.linalg.eigh(Sigma)
    inverse = 1 / values
    traces = [(-1) ** m * np.sum(inverse**m) for m in range(r + 1)]
    total = 0.0
    for D in _differences(X, Y):
        squares = (D @ vectors) ** 2
        quads = [(-1) ** (m - 1) * squares @ inverse ** (m + 1) for m in range(r + 1)]
        total += np.sum(np.exp(-squares @ inverse / 2) * _moments(r, traces, quads)[r])
    return total / (len(X) * len(Y) * np.sqrt(np.prod(2 * np.pi * values)))


def _plugin_criterion(Z, r):
    """Plug-in estimate of the AMISE of ``H`` for the sphered data ``Z``."""
    n = len(Z)
    F, B, V, c = _bias_functional(Z, _pilot(Z, r, scv=False), r)

    def criterion(H):
        h, trace = H.ravel(), np.trace(H)
        squared_bias = (
            h @ F @ h
            + trace * np.sum(H * B)
            + trace**2 * c[0]
            + np.sum(H @ H * V)
            + np.sum(H * H) * c[1]
        )
        return _variance(H, n, r) + (-1) ** r * squared_bias / 4

    return criterion


def _scv_criterion(Z, r):
    """Smoothed cross-validation estimate of the MISE of ``H`` for the sphered data ``Z``."""
    n, d = Z.shape
    G = _pilot(Z, r, scv=True) ** 2 * np.eye(d)
    base = _normal_sum(Z, Z, 2 * G, r)

    def criterion(H):
        smoothed = _normal_sum(Z, Z, 2 * H + 2 * G, r) - 2 * _normal_sum(Z, Z, H + 2 * G, r)
        return _variance(H, n, r) + max((-1) ** r * (smoothed + base), 0.0)

    return criterion


def _variance(H, n, r):
    """Integrated variance ``n^-1 |H|^-1/2 2^-(d+r) pi^-d/2 nu_r(H^-1)``, inf for singular H."""
    values = np.linalg.eigvalsh(H)
    if values[0] <= 0:
        return np.inf
    d = len(values)
    nu = _moments(r, [np.sum(values ** (-m)) for m in range(r + 1)], [0.0] * (r + 1))[r]
    return nu / (n * np.sqrt(np.prod(values)) * 2 ** (d + r) * np.pi ** (d / 2))


def _bias_functional(Z, g, r):
    """Terms of ``Q(H) = n^-2 sum_ij phi_G(Z_i - Z_j) E[(u'u)^r (u'Hu)^2]`` for ``G = g^2 I``.

    With ``u ~ N(z, -a I)``, ``a = g^-2`` and ``z = a (Z_i - Z_j)``,
    ``E[(u'u)^r (u'Hu)^2]`` is ``alpha (z'Hz)^2 + beta tr(H) z'Hz +
    gamma tr(H)^2 + delta z'H^2 z + epsilon tr(H^2)`` with coefficients that
    are polynomials in ``z'z``, so ``Q`` is a quadratic form in ``H`` whose
    terms are summed once. Returns ``F`` (on ``vec H``), ``B``, ``V`` and
    ``c = (sum gamma, sum epsilon)``, weighted by the pilot density.
    """
    d = Z.shape[1]
    a = g**-2
    F, B, V, c = np.zeros((d * d, d * d)), np.zeros((d, d)), np.zeros((d, d)), np.zeros(2)
    for D in _differences(Z, Z):
        s = np.sum(D**2, axis=1)
        alpha, beta, gamma, delta, epsilon = _bias_coefficients(a**2 * s, a, d, r)
        weight = np.exp(-a * s / 2)
        z = a * D
        zz = (z[:, :, None] * z[:, None, :]).reshape(len(z), -1)
        F += zz.T @ (zz * (weight * alpha)[:, None])
        B += z.T @ (z * (weight * beta)[:, None])
        V += z.T @ (z * (weight * delta)[:, None])
        c += [np.sum(weight * gamma), np.sum(weight * epsilon)]
    scale = (a / (2 * np.pi)) ** (d / 2) / len(Z) ** 2
    return F * scale, B * scale, V * scale, c * scale


def _bias_coefficients(s, a, d, r):
    """Coefficients of ``E[(u'u)^r (u'Hu)^2]``, ``u ~ N(z, -a I)``, as functions of ``s = z'z``.

    The moment sums, over the set partitions of the ``r + 2`` forms, the
    products of their joint cumulants. With ``C = -a I`` the cumulant of a
    block of ``m`` forms containing ``h`` copies of ``H`` is
    ``c_m (-a tr(H^h) + m z'H^h z)``, ``c_m = 2^(m-1) (m-1)! (-a)^(m-1)``.
    The forms ``u'u`` outside the blocks of ``H`` contribute their moments.
    """
    mu = _moments(
        r, [d * (-a) ** m for m in range(r + 1)], [(-a) ** (m - 1) * s for m in range(r + 1)]
    )
    cm = [0.0] + [2 ** (m - 1) * math.factorial(m - 1) * (-a) ** (m - 1) for m in range(1, r + 3)]
    alpha = beta = gamma = delta = epsilon = 0.0
    for j in range(r + 1):  # both forms of H in one block with j forms u'u
        term = math.comb(r, j) * cm[j + 2] * mu[r - j]
        delta, epsilon = delta + (j + 2) * term, epsilon - a * term
        for i in range(r + 1 - j):  # separate blocks with i and j forms u'u
            ways = math.factorial(r) // (
                math.factorial(i) * math.factorial(j) * math.factorial(r - i - j)
            )
            term = ways * cm[i + 1] * cm[j + 1] * mu[r - i - j]
            alpha = alpha + (i + 1) * (j + 1) * term
            beta = beta - a * (i + j + 2) * term
            gamma = gamma + a**2 * term
    return alpha, beta, gamma, delta, epsilon


def _odd_factorial(m):
    """``(m - 1)!!`` for even ``m``: the ``m``-th moment of a standard normal."""
    return math.prod(range(1, m, 2))


def _pilot(Z, r, scv):
    """Scalar pilot ``g`` for the sphered data ``Z``, R's ``ks:::gdscalar`` with two stages.

    ``g`` minimizes the asymptotic mean squared error of the estimate of
    ``(vec I_{d^(r+2)})' psi_{2r+4}`` (Chacón & Duong 2013), whose bias needs
    ``psi_{2r+6}``, estimated in turn with a normal reference bandwidth.
    """
    n, d = Z.shape
    of4, of6 = _odd_factorial(2 * r + 4), _odd_factorial(2 * r + 6)
    nu2 = math.prod(d + 2 * j for j in range(r + 2))  # E[(chi^2_d)^(r+2)]
    nu4 = math.prod(d + 2 * j for j in range(r + 4))
    m = 2 * r + 6  # the normal reference of psi_{2r+8}, contracted once, has squared norm b3
    b3 = (
        (4 * np.pi) ** -d
        * 2.0 ** -(m + 2)
        * math.prod(d + 2 * j for j in range(m // 2 + 1))
        * (_odd_factorial(m + 2) + (d - 1) * _odd_factorial(m))
    )
    if scv:
        b1 = 2.0 ** (-2 * r - 5) * (4 * np.pi) ** -d * of6 * nu4
        b2 = -(d + 2 * r + 4) * 2.0 ** (-2 * r - 6) * (4 * np.pi) ** -d * of6 * nu4
        b3, cf = 4 * b3, (2.0**-d, 2 ** (1 - d / 2), 4)
    else:
        b1 = 2 * (2 * np.pi) ** -d * of6 * nu4
        b2 = -(d + 2 * r + 4) * 2 ** (2 - r - d / 2) * (2 * np.pi) ** -d * of6 * nu4
        cf = (1, 1, 1)
    g6 = (2 * b1 / (-b2 + np.sqrt(b2**2 + 4 * b1 * b3)) / n) ** (1 / (d + 2 * r + 8))
    eta = _normal_sum(Z, Z, g6**2 * np.eye(d), r + 3)
    l0 = (2 * np.pi) ** (-d / 2)
    a1 = cf[0] * (2 * d + 4 * r + 8) * l0**2 * of4 * nu2
    a2 = cf[1] * (d + 2 * r + 2) * l0 * of4 * eta
    a3 = cf[2] * eta**2
    return (2 * a1 / ((-a2 + np.sqrt(a2**2 + 4 * a1 * a3)) * n)) ** (1 / (d + 2 * r + 6))


def _minimize(criterion, start):
    """Minimize ``criterion(H)`` over ``H = L L'`` from ``start``, as ks with BFGS."""
    d = len(start)
    rows, cols = np.tril_indices(d)

    def matrix(theta):
        L = np.zeros((d, d))
        L[rows, cols] = theta
        return L @ L.T

    scale = criterion(start)
    theta = np.linalg.cholesky(start)[rows, cols]
    result = optimize.minimize(lambda t: criterion(matrix(t)) / scale, theta, method="BFGS")
    return matrix(result.x)


def _functional_bandwidth(X):
    """``h^2 S`` minimizing the AMSE of ``n^-2 sum_ij phi_H(X_i - X_j)``, as ks's ``Hpi.kfe``.

    The AMSE vanishes where ``n^-1 |H|^-1/2 phi(0) + tr(H Psi_2) / 2 = 0``;
    ``Psi_2`` is estimated on the sphered data with the normal reference pilot.
    """
    Z, _ = _sphere(X)
    n, d = Z.shape
    g = (2 ** (d / 2 + 1) / (n * (d + 2))) ** (1 / (d + 4))
    laplacian = _normal_sum(Z, Z, g**2 * np.eye(d), 1)
    h2 = (-2 * (2 * np.pi) ** (-d / 2) / (n * laplacian)) ** (2 / (d + 2))
    return h2 * np.cov(X, rowvar=False).reshape(d, d)


def _gradient_variance(X):
    """``grad f(mean)' S grad f(mean)`` with the normal scale bandwidth for the gradient."""
    n, d = X.shape
    S = np.cov(X, rowvar=False).reshape(d, d)
    H = (4 / (n * (d + 4))) ** (2 / (d + 6)) * S
    grad = KernelDensity(H).fit(X).gradient(X.mean(axis=0, keepdims=True))[0]
    return grad @ S @ grad
