"""Robust logistic and Poisson regression.

The estimators follow the scikit-learn API. The logistic models are classifiers
(``predict_proba``, ``decision_function``, ``predict``) and the Poisson models
are regressors whose ``predict`` returns the fitted mean. All of them set
``coef_``, ``intercept_``, ``cov_params_``, ``std_errors_``, ``z_values_``,
``p_values_`` and the robustness ``weights_`` of the observations, and print a
coefficient table with ``summary()``.
"""

import warnings
from functools import lru_cache

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import CubicHermiteSpline
from scipy.special import expit, log1p, ndtr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_generator, check_count, check_option, check_range
from .covariance import MCD
from .regression import _bisquare, _coefficient_table, _design, _elemental_subset, _split
from .scale import _bisquare_weights, _rho

__all__ = [
    "BYLogisticRegression",
    "MTPoissonRegressor",
    "MallowsLogisticRegression",
    "MallowsPoissonRegressor",
]

_MAX_ETA = 30.0  # linear predictors are clipped to +-30 before exp


class _RobustGLM(BaseEstimator):
    """Base class: the design matrix, inference and the coefficient table."""

    def _validated_design(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_samples=2)
        if isinstance(self, _LogisticMixin):
            self._binary(y)
        design = _design(X, self.fit_intercept)
        n, p = design.shape
        if n <= p:
            raise ValueError(f"needs more observations than coefficients (got n={n}, p={p})")
        if np.linalg.matrix_rank(design) < p:
            raise ValueError("X is singular: its columns (and the intercept) are collinear")
        return X, design, y

    def _set_estimates(self, coef, cov, weights):
        self.intercept_, self.coef_ = _split(coef, self.fit_intercept)
        self.cov_params_ = cov
        self.std_errors_ = np.sqrt(np.diag(cov))
        self.z_values_ = coef / self.std_errors_
        self.p_values_ = 2 * stats.norm.sf(np.abs(self.z_values_))
        self.weights_ = weights

    def _linear_predictor(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_

    def summary(self, feature_names=None):
        """Coefficient table as R's ``summary(glmrob)``.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names of the columns of ``X``; ``x0, x1, ...`` by default.

        Returns
        -------
        str
            Estimates, standard errors, z values and two-sided normal p-values.
        """
        check_is_fitted(self)
        return _coefficient_table(self, feature_names, "z", self.z_values_)


class _LogisticMixin(ClassifierMixin):
    """Binary targets and the classifier methods."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _binary(self, y):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported. "
                f"needs 2 classes, got {len(self.classes_)}"
            )
        return y.astype(float)

    def decision_function(self, X):
        """Linear predictor ``X @ coef_ + intercept_`` (log odds of ``classes_[1]``)."""
        return self._linear_predictor(X)

    def predict_proba(self, X):
        """Probabilities of ``classes_[0]`` and ``classes_[1]``, one row per observation."""
        probability = expit(self._linear_predictor(X))
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        """Most probable class."""
        linear = self._linear_predictor(X)
        return self.classes_[(linear > 0).astype(int)]


class _PoissonMixin(RegressorMixin):
    """Count targets and the fitted mean."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.positive_only = True
        return tags

    def _counts(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"expected a 1-D response, got shape {y.shape}")
        if not np.isfinite(y).all():
            raise ValueError("the response must be finite")
        if y.min() < 0:
            raise ValueError("the response of a Poisson model must be nonnegative")
        if self.fit_intercept and not np.any(y > 0):
            raise ValueError("an all-zero response has no finite Poisson intercept")
        return y

    def predict(self, X):
        """Fitted mean ``exp(X @ coef_ + intercept_)``."""
        return _poisson_mean(self._linear_predictor(X))


class _MallowsGLM(_RobustGLM):
    """Mallows quasi-likelihood estimator of Cantoni & Ronchetti (2001)."""

    def __init__(self, *, c=1.345, x_weights="none", fit_intercept=True, max_iter=100, tol=1e-8):
        self.c = c
        self.x_weights = x_weights
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        check_range("c", self.c, 0, np.inf, low_open=True, high_open=True)
        check_count("max_iter", self.max_iter)
        check_range("tol", self.tol, 0, np.inf, low_open=True, high_open=True)
        X, design, y = self._validated_design(X, y)
        y = self._response(y)
        w = _x_weights(self.x_weights, X, design)
        fit = _mallows(design, y, self._family, self.c, w, self.max_iter, self.tol)
        coef, cov, residual_weights, self.n_iter_, converged = fit
        self.converged_ = converged
        if not converged:
            warnings.warn("Mallows estimator did not converge", RuntimeWarning, 2)
        self._set_estimates(coef, cov, residual_weights * w)
        return self


_MALLOWS_DOC = """

    Solves ``sum((psi_c(r_i) - E[psi_c(r_i)]) w(x_i) sqrt(V(mu_i)) x_i) = 0``,
    with ``r_i = (y_i - mu_i) / sqrt(V(mu_i))`` the Pearson residuals,
    ``psi_c`` Huber's function and ``E[psi_c(r_i)]`` its expectation under the
    model, which makes the estimator Fisher consistent. Large residuals have
    bounded influence; the weights ``w(x_i)`` bound the influence of leverage
    points. Fisher scoring starts from the maximum likelihood fit, as R's
    ``glmrob(method = "Mqle")``. The estimator bounds the influence function,
    but its breakdown point is not high.

    Parameters
    ----------
    c : float, default 1.345
        Huber tuning constant; larger values are closer to maximum likelihood.
    x_weights : {{"none", "hat", "mcd"}}, default "none"
        Weights on the regressors. ``"hat"`` is ``sqrt(1 - h_ii)`` with
        ``h_ii`` the leverages (Cantoni & Ronchetti 2001; robustbase computes
        ``(1 - h_ii)^2`` under the same name). ``"mcd"`` is
        ``1 / sqrt(1 + 8 max(0, (d_i^2 - p) / sqrt(2 p)))`` with ``d_i`` robust
        distances from a deterministic :class:`~robustmodels.covariance.MCD`,
        as robustbase's ``"covMcd"``; it needs continuous regressors.
    fit_intercept : bool, default True
        Add an intercept.
    max_iter : int, default 100
        Maximum Fisher scoring steps; a ``RuntimeWarning`` reports non-convergence.
    tol : float, default 1e-8
        Relative change of the coefficients at which the iterations stop.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slopes.
    intercept_ : float
        Intercept; 0 without ``fit_intercept``.
    cov_params_ : ndarray of shape (p, p)
        Asymptotic covariance matrix of ``[intercept_, *coef_]``:
        ``M^-1 Q M^-1 / n`` with ``Q`` the variance of the estimating function.
        Cantoni & Ronchetti (and robustbase) subtract the squared mean of the
        correction terms instead of each squared term, which gives slightly
        larger standard errors.
    std_errors_, z_values_, p_values_ : ndarray of shape (p,)
        Standard errors, z statistics and two-sided normal p-values.
    weights_ : ndarray of shape (n_samples,)
        Robustness weights ``min(1, c / |r_i|) w(x_i)``.
    n_iter_ : int
        Fisher scoring steps.{extra}

    References
    ----------
    Cantoni, E. and Ronchetti, E. (2001). Robust inference for generalized
    linear models. *Journal of the American Statistical Association*, 96(455),
    1022-1030.

    Examples
    --------
{example}
    """


class MallowsLogisticRegression(_LogisticMixin, _MallowsGLM):
    __doc__ = "Robust logistic regression by Mallows quasi-likelihood." + _MALLOWS_DOC.format(
        extra="\n    classes_ : ndarray of shape (2,)\n        Class labels.",
        example="""    >>> import numpy as np
    >>> from robustmodels.glm import MallowsLogisticRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> y = rng.binomial(1, 1 / (1 + np.exp(-1 - 2 * X[:, 0])))
    >>> y[np.argsort(X[:, 0])[-5:]] = 0  # five misclassified observations
    >>> fit = MallowsLogisticRegression().fit(X, y)
    >>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
    (0.76, [1.64])""",
    )
    _family = "binomial"

    def _response(self, y):
        return self._binary(y)


class MallowsPoissonRegressor(_PoissonMixin, _MallowsGLM):
    __doc__ = "Robust Poisson regression by Mallows quasi-likelihood." + _MALLOWS_DOC.format(
        extra="",
        example="""    >>> import numpy as np
    >>> from robustmodels.glm import MallowsPoissonRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> y = rng.poisson(np.exp(1 + 0.5 * X[:, 0]))
    >>> y[:10] = 40  # ten outliers
    >>> fit = MallowsPoissonRegressor().fit(X, y)
    >>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
    (1.07, [0.54])""",
    )
    _family = "poisson"

    def _response(self, y):
        return self._counts(y)


class BYLogisticRegression(_LogisticMixin, _RobustGLM):
    """Bianco-Yohai estimator of logistic regression (Croux & Haesbroeck 2003).

    Minimizes ``sum(rho(d_i) + G(F_i) + G(1 - F_i))``, with ``d_i`` the
    deviance of observation ``i``, ``F_i`` its fitted probability and ``G`` the
    correction that makes the estimator Fisher consistent (Bianco & Yohai
    1996). ``rho`` is the function of Croux & Haesbroeck, linear up to ``c``
    and with derivative ``exp(-sqrt(t))`` after it, for which the estimator
    exists whenever maximum likelihood does. With ``weighted=True`` (the
    weighted Bianco-Yohai estimator) the sum runs over the observations whose
    robust distance in the ``X`` space is at most the 0.975 chi-square
    quantile, which also protects against bad leverage points; it needs
    continuous regressors. R's ``glmrob(method = "WBY")`` uses those weights
    only for the starting value and then minimizes the unweighted criterion.

    Parameters
    ----------
    c : float, default 0.5
        Tuning constant of ``rho``.
    weighted : bool, default True
        Leave out the leverage points flagged by a deterministic
        :class:`~robustmodels.covariance.MCD` with ``alpha=0.75``.
    fit_intercept : bool, default True
        Add an intercept.
    max_iter : int, default 1000
        Maximum BFGS iterations; a ``RuntimeWarning`` reports non-convergence.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slopes.
    intercept_ : float
        Intercept; 0 without ``fit_intercept``.
    cov_params_ : ndarray of shape (p, p)
        Asymptotic (sandwich) covariance matrix of ``[intercept_, *coef_]``.
    std_errors_, z_values_, p_values_ : ndarray of shape (p,)
        Standard errors, z statistics and two-sided normal p-values.
    weights_ : ndarray of shape (n_samples,)
        ``psi(d_i) / psi(0)``, the weight of each deviance, and 0 for the
        leverage points left out.
    objective_ : float
        Mean of the minimized criterion.
    classes_ : ndarray of shape (2,)
        Class labels.

    References
    ----------
    Bianco, A. M. and Yohai, V. J. (1996). Robust estimation in the logistic
    regression model. In *Robust Statistics, Data Analysis, and Computer
    Intensive Methods*, Lecture Notes in Statistics 109, 17-34. Springer.

    Croux, C. and Haesbroeck, G. (2003). Implementing the Bianco and Yohai
    estimator for logistic regression. *Computational Statistics & Data
    Analysis*, 44(1-2), 273-295.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.glm import BYLogisticRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> y = rng.binomial(1, 1 / (1 + np.exp(-1 - 2 * X[:, 0])))
    >>> X[:5], y[:5] = 8, 0  # five bad leverage points
    >>> fit = BYLogisticRegression().fit(X, y)
    >>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
    (0.91, [1.88])
    """

    def __init__(self, *, c=0.5, weighted=True, fit_intercept=True, max_iter=1000):
        self.c = c
        self.weighted = weighted
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        check_range("c", self.c, 0, np.inf, low_open=True, high_open=True)
        check_count("max_iter", self.max_iter)
        X, design, y = self._validated_design(X, y)
        y = self._binary(y)
        kept = _regular_points(X) if self.weighted else np.ones(len(y), bool)
        if len(np.unique(y[kept])) < 2 or np.linalg.matrix_rank(design[kept]) < design.shape[1]:
            raise ValueError("weighted subset must contain both classes and a full-rank design")
        coef, cov, self.objective_, self.n_iter_, converged = _bianco_yohai(
            design[kept], y[kept], self.c, self.max_iter
        )
        self.converged_ = converged
        if not converged:
            warnings.warn("the Bianco-Yohai estimator did not converge", RuntimeWarning, 2)
        weights = kept * _by_psi(_deviance(design @ coef, y), self.c) / np.exp(-np.sqrt(self.c))
        self._set_estimates(coef, cov, weights)
        return self


class MTPoissonRegressor(_PoissonMixin, _RobustGLM):
    """MT-estimator of Poisson regression (Valdora & Yohai 2014).

    The square root stabilizes the variance of a Poisson count, so the
    estimator is an M-estimator on that scale: it minimizes
    ``sum(w(x_i) rho(sqrt(y_i) - m(mu_i)))`` with ``rho`` Tukey's bisquare and
    ``m(mu)`` the value that minimizes ``E[rho(sqrt(Y) - m)]`` for ``Y``
    Poisson with mean ``mu``, which makes it Fisher consistent. Because ``rho``
    is bounded, gross outliers get weight 0. The search starts from subsamples:
    each gives a fit, the half of the data with the smallest deviances is refit
    by maximum likelihood, and the candidate with the smallest criterion is
    improved with BFGS, as R's ``glmrob(method = "MT")``. ``m`` is a cubic
    Hermite spline with exact derivatives on the grid of robustbase, extended
    beyond a mean of 100 by ``sqrt(mu) + a / sqrt(mu)`` (robustbase switches to
    ``sqrt(mu)`` there, with a small jump). The search does not certify a
    global minimum.

    Parameters
    ----------
    c : float, default 2.1
        Bisquare tuning constant on the square-root scale, where the standard
        deviation is about 1/2.
    x_weights : {"none", "hat", "mcd"}, default "none"
        Weights on the regressors, as in :class:`MallowsPoissonRegressor`.
    fit_intercept : bool, default True
        Add an intercept.
    n_trials : int, default 500
        Random starting subsamples.
    max_iter : int, default 500
        Maximum BFGS iterations; a ``RuntimeWarning`` reports non-convergence.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the subsamples.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slopes.
    intercept_ : float
        Intercept; 0 without ``fit_intercept``.
    cov_params_ : ndarray of shape (p, p)
        Asymptotic (sandwich) covariance matrix of ``[intercept_, *coef_]``.
    std_errors_, z_values_, p_values_ : ndarray of shape (p,)
        Standard errors, z statistics and two-sided normal p-values.
    weights_ : ndarray of shape (n_samples,)
        Bisquare weights of ``sqrt(y_i) - m(mu_i)`` times ``w(x_i)``.
    objective_ : float
        Minimized criterion.

    References
    ----------
    Valdora, M. and Yohai, V. J. (2014). Robust estimators for generalized
    linear models. *Journal of Statistical Planning and Inference*, 146, 31-48.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.glm import MTPoissonRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> y = rng.poisson(np.exp(1 + 0.5 * X[:, 0]))
    >>> y[:30] = 40  # thirty outliers
    >>> fit = MTPoissonRegressor(n_trials=50, random_state=0).fit(X, y)
    >>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
    (0.97, [0.57])
    >>> int((fit.weights_[:30] == 0).sum())
    30
    """

    def __init__(
        self,
        *,
        c=2.1,
        x_weights="none",
        fit_intercept=True,
        n_trials=500,
        max_iter=500,
        random_state=None,
    ):
        self.c = c
        self.x_weights = x_weights
        self.fit_intercept = fit_intercept
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        check_range("c", self.c, 0, np.inf, low_open=True, high_open=True)
        check_count("n_trials", self.n_trials)
        check_count("max_iter", self.max_iter)
        X, design, y = self._validated_design(X, y)
        y = self._counts(y)
        w = _x_weights(self.x_weights, X, design)
        problem = _MTProblem(design, y, w, float(self.c))
        start = problem.start(self.n_trials, as_generator(self.random_state))
        fit = optimize.minimize(
            problem.objective,
            start,
            jac=problem.gradient,
            method="BFGS",
            options={"maxiter": self.max_iter, "gtol": 1e-8 * len(y)},
        )
        self.n_iter_ = fit.nit
        self.converged_ = bool(
            fit.success or np.linalg.norm(problem.gradient(fit.x), ord=np.inf) <= 1e-8 * len(y)
        )
        if not self.converged_:
            warnings.warn("the MT estimator did not converge", RuntimeWarning, 2)
        self.objective_ = float(fit.fun)
        weights = w * _bisquare_weights(problem.residuals(fit.x), self.c)
        self._set_estimates(fit.x, problem.covariance(fit.x), weights)
        return self


def _x_weights(kind, X, design):
    """Weights that bound the influence of leverage points."""
    check_option("x_weights", kind, ("none", "hat", "mcd"))
    if kind == "none":
        return np.ones(len(X))
    if kind == "hat":
        q = np.linalg.qr(design)[0]
        return np.sqrt(np.maximum(1 - np.sum(q**2, axis=1), 0))
    p = X.shape[1]
    distances = MCD(algorithm="deterministic").fit(X).mahalanobis(X)
    return 1 / np.sqrt(1 + np.maximum(0, 8 * (distances - p) / np.sqrt(2 * p)))


def _regular_points(X):
    """Observations within the 0.975 chi-square quantile of a robust centre of ``X``."""
    mcd = MCD(alpha=0.75, algorithm="deterministic").fit(X)
    return mcd.mahalanobis(X) <= stats.chi2.ppf(0.975, X.shape[1])


def _poisson_mean(eta):
    return np.exp(np.clip(eta, -_MAX_ETA, _MAX_ETA))


def _mean(eta, family):
    return expit(np.clip(eta, -_MAX_ETA, _MAX_ETA)) if family == "binomial" else _poisson_mean(eta)


def _variance(mu, family):
    return mu * (1 - mu) if family == "binomial" else mu


def _solve(design, weights, score):
    """Solve ``(X' W X) step = X' score``; least squares if it is singular."""
    return np.linalg.lstsq(design.T @ (design * weights[:, None]), design.T @ score, rcond=None)[0]


def _maximum_likelihood(design, y, family, max_iter=50, tol=1e-10):
    """Maximum likelihood fit of a logistic or Poisson model by Fisher scoring."""
    start = np.log(y + 0.5) if family == "poisson" else np.log((y + 0.5) / (1.5 - y))
    coef = np.linalg.lstsq(design, start, rcond=None)[0]
    for _ in range(max_iter):
        mu = _mean(design @ coef, family)
        step = _solve(design, np.maximum(_variance(mu, family), 1e-12), y - mu)
        # Bound changes of the linear predictor to avoid numerical separation/overflow.
        step *= min(1.0, 5.0 / max(np.max(np.abs(design @ step)), 1e-300))
        coef = coef + step
        if np.linalg.norm(step) <= tol * max(1.0, np.linalg.norm(coef)):
            break
    return coef


def _mallows_moments(mu, c, family):
    """``E[psi]``, ``E[psi (y - mu)] / V`` and ``E[psi^2]`` of the Huberized Pearson residual."""
    root = np.sqrt(_variance(mu, family))
    if family == "binomial":
        one, zero = np.clip((1 - mu) / root, -c, c), np.clip(-mu / root, -c, c)
        return mu * one + (1 - mu) * zero, one - zero, mu * one**2 + (1 - mu) * zero**2
    # Poisson: residuals are clipped below j1 = floor(mu - c root) + 1 and above j2
    low, high = np.floor(mu - c * root), np.floor(mu + c * root)
    d_low, d_high = stats.poisson.pmf(low, mu), stats.poisson.pmf(high, mu)
    p_low, p_high = stats.poisson.cdf(low, mu), stats.poisson.cdf(high, mu)
    middle = mu * (stats.poisson.pmf(low - 1, mu) - d_low - stats.poisson.pmf(high - 1, mu))
    middle += mu * d_high + (p_high - d_high) - (p_low - d_low)
    mean = c * (1 - p_high - p_low) + mu * (d_low - d_high) / root
    return mean, c * (d_low + d_high) + middle / root, c**2 * (p_low + 1 - p_high) + middle


def _mallows(design, y, family, c, w, max_iter, tol):
    """Fisher scoring for the Mallows quasi-likelihood estimating equation.

    Returns (coef, covariance, residual weights, iterations, converged).
    """
    n = len(y)
    coef = _maximum_likelihood(design, y, family)
    converged = False
    for iteration in range(1, max_iter + 1):  # noqa: B007 (used after the loop)
        mu = _mean(design @ coef, family)
        root = np.sqrt(np.maximum(_variance(mu, family), 1e-12))
        mean, slope, _ = _mallows_moments(mu, c, family)
        score = (np.clip((y - mu) / root, -c, c) - mean) * w * root
        step = _solve(design, slope * w * root**3, score)
        # Bound changes of the linear predictor to avoid numerical separation/overflow.
        step *= min(1.0, 5.0 / max(np.max(np.abs(design @ step)), 1e-300))
        coef = coef + step
        converged = np.linalg.norm(step) <= tol * max(tol, np.linalg.norm(coef))
        if converged:
            break
    mu = _mean(design @ coef, family)
    root = np.sqrt(np.maximum(_variance(mu, family), 1e-12))
    mean, slope, square = _mallows_moments(mu, c, family)
    bread = np.linalg.pinv(design.T @ (design * (slope * w * root**3)[:, None]) / n)
    meat = design.T @ (design * ((square - mean**2) * (w * root) ** 2)[:, None]) / n
    residuals = np.abs(y - mu) / root
    weights = np.minimum(1, c / np.maximum(residuals, 1e-300))
    return coef, bread @ meat @ bread / n, weights, iteration, converged


def _deviance(s, y):
    """Minus the log-likelihood of a binary ``y`` with log odds ``s``."""
    return log1p(np.exp(-np.abs(s))) + np.abs(s) * ((y - 0.5) * s < 0)


def _by_psi(t, c):
    """Derivative of the Croux-Haesbroeck ``rho``."""
    return np.exp(-np.sqrt(np.maximum(t, c)))


def _by_dpsi(t, c):
    root = np.sqrt(np.maximum(t, c))
    return np.where(t > c, -np.exp(-root) / (2 * root), 0.0)


def _by_rho(t, c):
    root = np.sqrt(np.maximum(t, c))
    beyond = np.exp(-np.sqrt(c)) * (2 + 2 * np.sqrt(c) + c) - 2 * np.exp(-root) * (1 + root)
    return np.where(t > c, beyond, t * np.exp(-np.sqrt(c)))


def _by_correction(s, c):
    """``G(F(s))``, with ``G(t)`` the integral of ``psi(-log u)`` over ``(0, t)``."""
    root = np.sqrt(np.maximum(_deviance(s, 1), c))
    tail = np.exp(0.25) * np.sqrt(np.pi) * (ndtr(np.sqrt(2) * (0.5 + root)) - 1)
    return expit(s) * np.exp(-root) + tail


def _by_loss(s, y, c):
    """Bianco-Yohai criterion of each observation, and its first two derivatives in ``s``."""
    probability = expit(s)
    spread = probability * (1 - probability)
    deviance = _deviance(s, y)
    psi, up, down = _by_psi(deviance, c), _by_psi(_deviance(s, 1), c), _by_psi(_deviance(-s, 1), c)
    loss = _by_rho(deviance, c) + _by_correction(s, c) + _by_correction(-s, c)
    first = -psi * (y - probability) + spread * (up - down)
    second = _by_dpsi(deviance, c) * (probability - y) ** 2 + spread * psi
    second += spread * (1 - 2 * probability) * (up - down)
    second -= spread * (
        _by_dpsi(_deviance(s, 1), c) * (1 - probability)
        + _by_dpsi(_deviance(-s, 1), c) * probability
    )
    return loss, first, second


def _bianco_yohai(design, y, c, max_iter):
    """Minimize the Bianco-Yohai criterion from the maximum likelihood fit.

    Returns (coef, covariance, objective, iterations, converged).
    """
    n = len(y)

    def objective(coef):
        loss, first, _ = _by_loss(design @ coef, y, c)
        return loss.mean(), design.T @ first / n

    start = _maximum_likelihood(design, y, "binomial")
    fit = optimize.minimize(
        objective, start, jac=True, method="BFGS", options={"maxiter": max_iter, "gtol": 1e-9}
    )
    _, first, second = _by_loss(design @ fit.x, y, c)
    bread = np.linalg.pinv(design.T @ (design * second[:, None]) / n)
    meat = design.T @ (design * first[:, None] ** 2) / n
    converged = bool(fit.success or np.linalg.norm(design.T @ first / n, ord=np.inf) <= 1e-8)
    return fit.x, bread @ meat @ bread.T / n, float(fit.fun), max(1, fit.nit), converged


class _MTMean:
    """``m(mu)``: the minimizer of ``E[rho(sqrt(Y) - m)]`` for ``Y`` Poisson with mean ``mu``.

    Computed on a grid up to 100 with its derivative (by implicit
    differentiation of ``E[psi(sqrt(Y) - m)] = 0``) and interpolated; beyond
    the grid it is ``sqrt(mu) + a / sqrt(mu)``, continuous at 100.
    """

    def __init__(self, c):
        grid = np.r_[np.arange(30) / 10, np.arange(3.0, 101.0)]
        values = np.array([self._solve(mu, c) for mu in grid])
        self.spline = CubicHermiteSpline(grid, *values.T)
        self.tail = (values[-1, 0] - 10) * 10

    @staticmethod
    def _solve(mu, c):
        counts = np.arange(stats.poisson.isf(1e-14, mu) + 3)
        mass, roots = stats.poisson.pmf(counts, mu), np.sqrt(counts)
        m = 0.0
        if mu > 0:
            # A small tuning constant makes the expected loss multimodal.
            # Bracket every sampled local minimum, including the endpoint 0,
            # rather than assuming that a single bounded search is unimodal.
            grid = np.sort(np.r_[roots, (roots[:-1] + roots[1:]) / 2])
            losses = _rho(roots[:, None] - grid, c).T @ mass
            candidates = [(float(losses[0]), 0.0)]
            minima = np.flatnonzero((losses[1:-1] < losses[:-2]) & (losses[1:-1] <= losses[2:])) + 1
            for i in minima:
                fit = optimize.minimize_scalar(
                    lambda m: mass @ _rho(roots - m, c),
                    bounds=(grid[i - 1], grid[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-12},
                )
                candidates.append((float(fit.fun), float(fit.x)))
            m = min(candidates)[1]
        psi, dpsi, _ = _bisquare(roots - m, c)
        return m, psi @ (np.r_[0, mass[:-1]] - mass) / (dpsi @ mass)

    def __call__(self, mu, nu=0):
        inside = mu <= 100
        root = np.sqrt(np.where(inside, 100.0, mu))
        tails = [
            root + self.tail / root,
            0.5 / root - 0.5 * self.tail / root**3,
            -0.25 / root**3 + 0.75 * self.tail / root**5,
        ]
        return np.where(inside, self.spline(np.minimum(mu, 100), nu), tails[nu])


@lru_cache(maxsize=8)
def _mt_mean(c):
    return _MTMean(c)


class _MTProblem:
    """Criterion, gradient, starting value and covariance of the MT-estimator."""

    def __init__(self, design, y, w, c):
        self.design, self.w, self.c = design, w, c
        self.y, self.root = y, np.sqrt(y)
        self.m = _mt_mean(c)

    def residuals(self, coef):
        return self.root - self.m(_poisson_mean(self.design @ coef))

    def objective(self, coef):
        return float(self.w @ _rho(self.residuals(coef), self.c))

    def _terms(self, coef):
        """``psi``, ``psi'`` (of the normalized rho), ``mu``, ``m'`` and ``m''``."""
        mu = _poisson_mean(self.design @ coef)
        _, dpsi, psi = _bisquare(self.root - self.m(mu), self.c)
        return psi, 6 * dpsi / self.c**2, mu, self.m(mu, 1), self.m(mu, 2)

    def gradient(self, coef):
        psi, _, mu, slope, _ = self._terms(coef)
        return -self.design.T @ (self.w * psi * slope * mu)

    def start(self, n_trials, rng):
        """Best trimmed maximum likelihood fit over random elemental subsamples."""
        design, y = self.design, self.y
        half = (len(y) + 1) // 2
        best, best_value = None, np.inf
        for _ in range(n_trials):
            subset = _elemental_subset(design, rng)
            coef = np.linalg.lstsq(design[subset], np.log(np.maximum(y[subset], 0.5)), rcond=None)[
                0
            ]
            eta = np.clip(design @ coef, -_MAX_ETA, _MAX_ETA)
            deviance = np.abs(y * (np.log(np.maximum(y, 1)) - eta) - (y - np.exp(eta)))
            kept = np.argpartition(deviance, half - 1)[:half]
            coef = _maximum_likelihood(design[kept], y[kept], "poisson")
            value = self.objective(coef)
            if value < best_value:
                best, best_value = coef, value
        return best

    def covariance(self, coef):
        psi, dpsi, mu, slope, curvature = self._terms(coef)
        n = len(mu)
        score = self.w * psi * slope * mu
        hessian = self.w * (-dpsi * (slope * mu) ** 2 + psi * curvature * mu**2 + psi * slope * mu)
        bread = np.linalg.pinv(self.design.T @ (self.design * hessian[:, None]) / n)
        meat = self.design.T @ (self.design * score[:, None] ** 2) / n
        return bread @ meat @ bread.T / n
