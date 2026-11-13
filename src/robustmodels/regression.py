"""Robust linear and nonlinear regression estimators and their fitting steps.

The estimators follow the scikit-learn API: ``fit(X, y)`` takes observations in
rows and sets ``coef_``, ``intercept_`` and ``scale_`` (robust scale of the
residuals); ``predict(X)`` returns ``X @ coef_ + intercept_``. Nonlinear M
regression stores ``params_`` and evaluates the supplied model callable.
"""

import warnings

import numpy as np
from scipy import optimize, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_generator, check_count, check_range
from .covariance import h_alpha_n, mcd_consistency
from .scale import (
    _bisquare_weights,
    _mscale_bisquare,
    _rho,
    _tuning_for_breakdown,
    _tuning_for_efficiency,
    mad,
)

__all__ = ["LTSRegressor", "MMRegressor", "NonlinearMRegressor", "SRegressor"]

# Residuals with |r / scale| above this get weight 0 (2.5% total, 1.25% per tail).
_CUTOFF = stats.norm.ppf(0.9875)


class _RobustRegressor(RegressorMixin, BaseEstimator):
    """Base class: input checks, the design matrix and ``predict``."""

    def _validated_design(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True, ensure_min_samples=2)
        design = _design(X, self.fit_intercept)
        n, p = design.shape
        if n <= p:
            raise ValueError(f"needs more observations than coefficients (got n={n}, p={p})")
        if np.linalg.matrix_rank(design) < p:
            raise ValueError("X is singular: its columns (and the intercept) are collinear")
        return design, y

    def predict(self, X):
        """Predicted values ``X @ coef_ + intercept_``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_


class LTSRegressor(_RobustRegressor):
    """Least Trimmed Squares regression (Rousseeuw 1984).

    Least squares fit to the ``h`` observations with the smallest sum of squared
    residuals, found with FAST-LTS (Rousseeuw & Van Driessen 2006). The residual
    scale is made consistent at the normal model; with outliers present, the
    consistency factors overestimate the error standard deviation, as in R's
    ``ltsReg``. The reweighting step then
    refits least squares to the observations whose standardized residual is at
    most 2.24 in absolute value. The raw estimator has breakdown point close to
    ``(n - h + 1) / n``, about 50% for ``alpha = 0.5``; its efficiency is low,
    and the reweighting step improves it. The search does not certify a global
    minimum. If ``h`` observations fit exactly, a ``RuntimeWarning`` reports it,
    the raw fit is kept and the scale is 0.

    Parameters
    ----------
    alpha : float in [0.5, 1], default 0.5
        Coverage. The subset size ``h = h_alpha_n(alpha, n, p)``, with ``p`` the
        number of coefficients including the intercept, goes from
        ``(n + p + 1) // 2`` at 0.5 to ``n`` (least squares) at 1.
    fit_intercept : bool, default True
        Add an intercept.
    n_trials : int, default 500
        Random starting subsets of FAST-LTS.
    reweighting : bool, default True
        Apply the reweighting step.
    small_sample_correction : bool, default True
        Multiply the scale by the factors of Pison, Van Aelst & Willems (2002).
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of FAST-LTS.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Final slopes.
    intercept_ : float
        Final intercept; 0 without ``fit_intercept``.
    scale_ : float
        Final robust scale of the residuals.
    support_ : ndarray of bool, shape (n_samples,)
        Observations used for the final fit.
    raw_coef_, raw_intercept_, raw_scale_ :
        Least trimmed squares fit and its corrected scale.
    raw_support_ : ndarray of bool, shape (n_samples,)
        The ``h``-subset.
    h_ : int
        Subset size.

    References
    ----------
    Rousseeuw, P. J. (1984). Least median of squares regression. *Journal of the
    American Statistical Association*, 79(388), 871-880.

    Rousseeuw, P. J. and Van Driessen, K. (2006). Computing LTS regression for
    large data sets. *Data Mining and Knowledge Discovery*, 12(1), 29-45.

    Pison, G., Van Aelst, S. and Willems, G. (2002). Small sample corrections for
    LTS and MCD. *Metrika*, 55, 111-123.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.regression import LTSRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 1))
    >>> y = 1 + 2 * X[:, 0] + rng.normal(scale=0.5, size=100)
    >>> y[:20] += 15  # twenty outliers
    >>> lts = LTSRegressor(random_state=0).fit(X, y)
    >>> round(lts.intercept_, 2), lts.coef_.round(2).tolist()
    (0.95, [2.04])
    >>> int(lts.support_[:20].sum())  # no outlier is used
    0
    """

    def __init__(
        self,
        *,
        alpha=0.5,
        fit_intercept=True,
        n_trials=500,
        reweighting=True,
        small_sample_correction=True,
        random_state=None,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_trials = n_trials
        self.reweighting = reweighting
        self.small_sample_correction = small_sample_correction
        self.random_state = random_state

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        check_count("n_trials", self.n_trials)
        design, y = self._validated_design(X, y)
        n, p = design.shape
        h = h_alpha_n(self.alpha, n, p)
        if h == n:
            subset = np.arange(n)
        else:
            subset = _fast_lts(design, y, h, self.n_trials, as_generator(self.random_state))
        self.h_ = h
        coef = self._raw_fit(design, y, subset)
        coef, self.scale_, self.support_ = self._reweighted_fit(design, y, coef)
        self.intercept_, self.coef_ = _split(coef, self.fit_intercept)
        return self

    def _raw_fit(self, design, y, subset):
        n, p = design.shape
        coef = _least_squares(design[subset], y[subset])
        residuals = y - design @ coef
        if self.h_ == n:  # least squares
            scale = np.sqrt(np.sum(residuals**2) / (n - p))
        else:
            trimmed = np.partition(residuals**2, self.h_ - 1)[: self.h_]
            scale = np.sqrt(trimmed.mean() * mcd_consistency(1, self.h_ / n))
            scale *= self._small_sample(p, n, False)
        self.raw_intercept_, self.raw_coef_ = _split(coef, self.fit_intercept)
        self.raw_scale_ = float(scale)
        self.raw_support_ = np.isin(np.arange(n), subset)
        return coef

    def _reweighted_fit(self, design, y, coef):
        scale = self.raw_scale_
        residuals = y - design @ coef
        tolerance = 1e-10 * np.ptp(y[self.raw_support_])  # the h-subset excludes outliers
        if scale <= tolerance:
            warnings.warn("exact fit: h observations lie on a hyperplane", RuntimeWarning, 3)
            return coef, 0.0, np.abs(residuals) <= tolerance
        if not self.reweighting:
            return coef, scale, self.raw_support_.copy()
        n, p = design.shape
        support = np.abs(residuals) <= _CUTOFF * scale
        coef = _least_squares(design[support], y[support])
        residuals = y[support] - design[support] @ coef
        kept = int(support.sum())
        # kept - 1, as robustbase: with the small-sample factors this is unbiased
        # at the normal model in simulations, while kept - p overestimates.
        scale = np.sqrt(np.sum(residuals**2) / (kept - 1))
        if kept < n:
            scale *= np.sqrt(mcd_consistency(1, kept / n)) * self._small_sample(p, n, True)
        return coef, float(scale), support

    def _small_sample(self, p, n, reweighted):
        if not self.small_sample_correction:
            return 1.0
        slopes = p - 1 if self.fit_intercept else p
        return _lts_small_sample(slopes, n, self.alpha, self.fit_intercept, reweighted)


class SRegressor(_RobustRegressor):
    """S-estimator of regression (Rousseeuw & Yohai 1984).

    Coefficients that minimize an M-estimator of the residual scale with Tukey's
    bisquare loss, found with FAST-S (Salibian-Barrera & Yohai 2006). The scale
    ``s`` solves ``sum(rho(r_i / s)) / (n - p) = breakdown``, with ``p`` the
    number of coefficients; the ``n - p`` denominator reduces small-sample bias
    (Maronna et al. 2019) and is also used by R's ``lmrob``. The
    bisquare constant makes the scale consistent at the normal model. At
    ``breakdown = 0.5`` the Gaussian efficiency is only about 29%; use
    :class:`MMRegressor` for efficient estimates. The search does not certify a
    global minimum. If more than about half of the residuals can be made zero,
    a ``RuntimeWarning`` reports the exact fit and the scale is 0.

    Parameters
    ----------
    breakdown : float in (0, 0.5], default 0.5
        Asymptotic breakdown point.
    fit_intercept : bool, default True
        Add an intercept.
    n_trials : int, default 500
        Random elemental starting subsets.
    n_best : int, default 5
        Candidates refined to convergence.
    k_steps : int, default 2
        Refinement steps applied to every start.
    max_iter : int, default 500
        Maximum refinement steps for the best candidates. Refinement stops when
        the coefficients change by less than ``1e-7`` relative, as in ``lmrob``;
        a ``RuntimeWarning`` reports non-convergence.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of FAST-S.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slopes.
    intercept_ : float
        Intercept; 0 without ``fit_intercept``.
    scale_ : float
        M-scale of the residuals.
    n_iter_ : int
        Refinement steps of the selected candidate; 0 for an exact fit.

    References
    ----------
    Rousseeuw, P. J. and Yohai, V. J. (1984). Robust regression by means of
    S-estimators. In *Robust and Nonlinear Time Series Analysis*, Lecture Notes
    in Statistics 26, 256-272. Springer.

    Salibian-Barrera, M. and Yohai, V. J. (2006). A fast algorithm for
    S-regression estimates. *Journal of Computational and Graphical
    Statistics*, 15(2), 414-427.

    Maronna, R. A., Martin, R. D., Yohai, V. J. and Salibian-Barrera, M. (2019).
    *Robust Statistics: Theory and Methods (with R)*, 2nd edition. Wiley.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.regression import SRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 1))
    >>> y = 1 + 2 * X[:, 0] + rng.normal(scale=0.5, size=100)
    >>> y[:20] += 15  # twenty outliers
    >>> s = SRegressor(random_state=0).fit(X, y)
    >>> round(s.intercept_, 2), s.coef_.round(2).tolist()
    (1.03, [2.04])
    """

    def __init__(
        self,
        *,
        breakdown=0.5,
        fit_intercept=True,
        n_trials=500,
        n_best=5,
        k_steps=2,
        max_iter=500,
        random_state=None,
    ):
        self.breakdown = breakdown
        self.fit_intercept = fit_intercept
        self.n_trials = n_trials
        self.n_best = n_best
        self.k_steps = k_steps
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        for name in ("n_trials", "n_best", "k_steps", "max_iter"):
            check_count(name, getattr(self, name))
        c = _tuning_for_breakdown(self.breakdown)
        design, y = self._validated_design(X, y)
        steps = (self.n_trials, self.n_best, self.k_steps, self.max_iter)
        fit = _fast_s(design, y, c, self.breakdown, steps, self.random_state)
        scale, coef, self.n_iter_, converged = fit
        _warn(scale, converged)
        self.scale_ = scale
        self.intercept_, self.coef_ = _split(coef, self.fit_intercept)
        return self


class MMRegressor(_RobustRegressor):
    """MM-estimator of regression (Yohai 1987).

    Starts from the :class:`SRegressor` fit and keeps its scale fixed while
    iteratively reweighted least squares lowers a bisquare loss with a larger
    tuning constant. This keeps the breakdown point of the S-estimator and
    reaches the requested Gaussian efficiency, as R's ``lmrob`` by default. The
    final loss is a local minimum near the S fit; it is not a global search.

    Parameters
    ----------
    breakdown : float in (0, 0.5], default 0.5
        Breakdown point of the initial S-estimator.
    efficiency : float in (0, 1), default 0.95
        Asymptotic Gaussian efficiency of the coefficients. The final tuning
        constant must not be smaller than the initial one; lower values raise
        ``ValueError``.
    fit_intercept : bool, default True
        Add an intercept.
    n_trials : int, default 500
        Random starting subsets of the S-estimator.
    max_iter : int, default 500
        Maximum steps of each iterative stage (S refinement and MM); a
        ``RuntimeWarning`` names the stage that did not converge.
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        Seed of the S-estimator.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Final slopes.
    intercept_ : float
        Final intercept; 0 without ``fit_intercept``.
    scale_ : float
        Residual scale of the S-estimator, kept fixed.
    s_coef_, s_intercept_ :
        Coefficients of the initial S-estimator.
    n_iter_ : int
        Iterations of the MM stage; 0 for an exact fit.
    cov_params_ : ndarray of shape (p, p)
        Asymptotic covariance matrix of the coefficients, ordered as
        ``[intercept_, *coef_]`` with ``fit_intercept`` (Croux, Dhaene &
        Hoorelbeke 2003, as ``lmrob``'s default). It is robust to
        heteroskedasticity; NaN for an exact fit.
    std_errors_, t_values_, p_values_ : ndarray of shape (p,)
        Standard errors, t statistics and two-sided p-values from a t
        distribution with ``n - p`` degrees of freedom, in the same order.

    References
    ----------
    Yohai, V. J. (1987). High breakdown-point and high efficiency robust
    estimates for regression. *The Annals of Statistics*, 15(2), 642-656.

    Croux, C., Dhaene, G. and Hoorelbeke, D. (2003). Robust standard errors for
    robust estimators. Discussion Paper 03.16, Center for Economic Studies,
    KU Leuven.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.regression import MMRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 1))
    >>> y = 1 + 2 * X[:, 0] + rng.normal(scale=0.5, size=100)
    >>> y[:20] += 15  # twenty outliers
    >>> mm = MMRegressor(random_state=0).fit(X, y)
    >>> round(mm.intercept_, 2), mm.coef_.round(2).tolist()
    (0.96, [2.04])
    """

    def __init__(
        self,
        *,
        breakdown=0.5,
        efficiency=0.95,
        fit_intercept=True,
        n_trials=500,
        max_iter=500,
        random_state=None,
    ):
        self.breakdown = breakdown
        self.efficiency = efficiency
        self.fit_intercept = fit_intercept
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        check_count("n_trials", self.n_trials)
        check_count("max_iter", self.max_iter)
        c0 = _tuning_for_breakdown(self.breakdown)
        c = _tuning_for_efficiency(self.efficiency)
        if c < c0:
            raise ValueError(
                "efficiency is too low for this breakdown: MM requires the final "
                "bisquare tuning constant >= the initial one"
            )
        design, y = self._validated_design(X, y)
        steps = (self.n_trials, 5, 2, self.max_iter)
        scale, coef, _, converged = _fast_s(design, y, c0, self.breakdown, steps, self.random_state)
        self.s_intercept_, self.s_coef_ = _split(coef, self.fit_intercept)
        _warn(scale, converged)
        self.n_iter_ = 0
        n, p = design.shape
        cov = np.full((p, p), np.nan)  # not estimable for an exact fit
        if scale > 0:
            s_residuals = y - design @ coef
            coef, self.n_iter_, converged = _mm_steps(design, y, coef, scale, c, self.max_iter)
            if not converged:
                warnings.warn("MM did not converge within max_iter", RuntimeWarning, 2)
            residuals = y - design @ coef
            cov = _mm_covariance(design, residuals, s_residuals, scale, (c0, c), self.breakdown)
        self.scale_ = scale
        self.intercept_, self.coef_ = _split(coef, self.fit_intercept)
        self.cov_params_ = cov
        self.std_errors_ = np.sqrt(np.diag(cov))
        self.t_values_ = coef / self.std_errors_
        self.p_values_ = 2 * stats.t.sf(np.abs(self.t_values_), n - p)
        return self

    def summary(self, feature_names=None):
        """Coefficient table as R's ``summary(lmrob)``.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names of the columns of ``X``; ``x0, x1, ...`` by default.

        Returns
        -------
        str
            Estimates, standard errors, t values and two-sided p-values, then
            the residual scale.
        """
        check_is_fitted(self)
        table = _coefficient_table(self, feature_names, "t", self.t_values_)
        return f"{table}\nResidual scale: {self.scale_:.4g}"


class NonlinearMRegressor(RegressorMixin, BaseEstimator):
    """Huber M-estimator of nonlinear regression.

    Fits ``y = model(X, *params) + error`` by iteratively reweighted nonlinear
    least squares: the residual scale is the median absolute residual divided
    by 0.6745 and the weights are Huber's ``min(1, c / |r_i / scale|)``, as
    R's ``nlrob`` by default. Large residuals have bounded influence, which
    protects against outliers in ``y``; outliers in ``X`` (leverage points)
    can still dominate the fit, and the result depends on the starting value.

    Parameters
    ----------
    model : callable
        ``model(X, *params)`` returns the fitted values, of shape (n_samples,).
    start : array-like of shape (n_params,)
        Starting value of the parameters.
    c : float, default 1.345
        Huber tuning constant, in units of the residual scale.
    max_iter : int, default 50
        Maximum reweighting steps; a ``RuntimeWarning`` reports non-convergence.
    tol : float, default 1e-6
        Relative change of the residuals at which the iterations stop.

    Attributes
    ----------
    params_ : ndarray of shape (n_params,)
        Estimated parameters.
    scale_ : float
        Residual scale.
    weights_ : ndarray of shape (n_samples,)
        Huber weights of the observations.
    cov_params_ : ndarray of shape (n_params, n_params)
        Asymptotic covariance matrix (Huber 1981):
        ``scale_^2 (sum(psi^2) / (n - p)) / mean(psi')^2 (J'J)^-1``, with ``J``
        the Jacobian of the model. ``nlrob`` uses the weighted ``J'WJ`` and the
        mean squared weight instead, which is not the asymptotic variance.
    std_errors_ : ndarray of shape (n_params,)
        Standard errors.
    n_iter_ : int
        Reweighting steps.

    References
    ----------
    Huber, P. J. (1981). *Robust Statistics*. Wiley.

    Examples
    --------
    >>> import numpy as np
    >>> from robustmodels.regression import NonlinearMRegressor
    >>> def decay(X, a, b):
    ...     return a * np.exp(-b * X[:, 0])
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 5, size=(80, 1))
    >>> y = decay(X, 10, 0.7) + rng.normal(scale=0.3, size=80)
    >>> y[:8] += 15  # eight outliers
    >>> fit = NonlinearMRegressor(decay, [5, 1]).fit(X, y)
    >>> fit.params_.round(2).tolist()
    [10.29, 0.71]
    """

    def __init__(self, model, start, *, c=1.345, max_iter=50, tol=1e-6):
        self.model = model
        self.start = start
        self.c = c
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """Fit to ``X`` of shape (n_samples, n_features) and ``y``. Returns ``self``."""
        check_range("c", self.c, 0, np.inf, low_open=True, high_open=True)
        check_range("tol", self.tol, 0, np.inf, low_open=True, high_open=True)
        check_count("max_iter", self.max_iter)
        if not callable(self.model):
            raise ValueError("model must be callable")
        params = np.asarray(self.start, dtype=float)
        if params.ndim != 1 or not params.size or not np.isfinite(params).all():
            raise ValueError("start must be a nonempty finite 1-D parameter vector")
        params = params.copy()
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True, ensure_min_samples=2)
        n, p = len(y), params.size
        if n <= p:
            raise ValueError(f"needs more observations than parameters (got n={n}, p={p})")

        def residuals(params):
            return y - self._evaluate(X, params)

        current = residuals(params)
        converged = False
        for iteration in range(1, self.max_iter + 1):  # noqa: B007 (used after the loop)
            scale = np.median(np.abs(current)) / 0.6745
            if scale == 0:
                converged = True
                break
            root = np.sqrt(_huber_weights(current / scale, self.c))
            fit = optimize.least_squares(lambda b: root * residuals(b), params)  # noqa: B023
            if not fit.success:
                raise RuntimeError(f"nonlinear least squares failed: {fit.message}")
            params = fit.x
            previous, current = current, residuals(params)
            change = np.linalg.norm(current - previous)
            converged = change <= self.tol * max(np.linalg.norm(previous), 1e-10)
            if converged:
                break
        if not converged:
            warnings.warn("the M-estimator did not converge within max_iter", RuntimeWarning, 2)
        scale = float(np.median(np.abs(current)) / 0.6745)
        self.params_, self.scale_, self.n_iter_ = params, scale, iteration
        self.converged_ = converged
        if scale == 0:
            warnings.warn("exact fit: most residuals are zero", RuntimeWarning, 2)
            self.weights_ = (current == 0).astype(float)
            self.cov_params_ = np.zeros((p, p))
        else:
            t = current / scale
            self.weights_ = _huber_weights(t, self.c)
            jacobian = optimize.approx_fprime(params, lambda b: self._evaluate(X, b))
            psi = np.clip(t, -self.c, self.c)
            factor = (
                scale**2 * (psi @ psi / (n - p)) / max(np.mean(np.abs(t) <= self.c), 1 / n) ** 2
            )
            self.cov_params_ = factor * np.linalg.pinv(jacobian.T @ jacobian)
        self.std_errors_ = np.sqrt(np.diag(self.cov_params_))
        return self

    def _evaluate(self, X, params):
        values = np.asarray(self.model(X, *params), dtype=float)
        if values.shape != (len(X),) or not np.isfinite(values).all():
            raise ValueError(
                "model must return one finite value per observation (shape (n_samples,))"
            )
        return values

    def predict(self, X):
        """Fitted values ``model(X, *params_)``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return self._evaluate(X, self.params_)


def _huber_weights(t, c):
    return np.minimum(1, c / np.maximum(np.abs(t), 1e-300))


def _coefficient_table(model, feature_names, statistic, values):
    """Estimates, standard errors, ``statistic`` values and p-values, one row per coefficient."""
    names = [f"x{j}" for j in range(model.n_features_in_)]
    if feature_names is not None:
        names = [str(name) for name in feature_names]
        if len(names) != model.n_features_in_:
            raise ValueError(f"expected {model.n_features_in_} feature names, got {len(names)}")
    estimates = model.coef_
    if model.fit_intercept:
        names, estimates = ["(Intercept)", *names], np.r_[model.intercept_, model.coef_]
    width = max(len(name) for name in names)
    head = f"{statistic} value"
    lines = [f"{'':{width}}  {'Estimate':>10}  {'Std. Error':>10}  {head:>8}  Pr(>|{statistic}|)"]
    rows = zip(names, estimates, model.std_errors_, values, model.p_values_, strict=True)
    for name, estimate, error, value, pvalue in rows:
        lines.append(f"{name:{width}}  {estimate:10.4g}  {error:10.4g}  {value:8.3f}  {pvalue:.3g}")
    return "\n".join(lines)


def _design(X, fit_intercept):
    """``X`` with a leading column of ones when fitting an intercept."""
    return np.column_stack([np.ones(len(X)), X]) if fit_intercept else X


def _split(coef, fit_intercept):
    """(intercept, slopes) from a coefficient vector of the design."""
    return (float(coef[0]), coef[1:]) if fit_intercept else (0.0, coef)


def _least_squares(design, y):
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _fast_lts(design, y, h, n_trials, rng):
    """Two C-steps per random elemental start, then converge the ten best subsets."""
    candidates = []
    for _ in range(n_trials):
        start = _elemental_subset(design, rng)
        coef = _least_squares(design[start], y[start])
        candidates.append(_lts_steps(design, y, coef, h, max_steps=2))
    candidates.sort(key=lambda candidate: candidate[0])
    finalists = [_lts_steps(design, y, coef, h) for _, _, coef in candidates[:10]]
    return min(finalists, key=lambda candidate: candidate[0])[1]


def _elemental_subset(design, rng):
    """Random p-subset, enlarged until its rows span the coefficient space."""
    n, p = design.shape
    order = rng.permutation(n)
    for k in range(p, n + 1):  # pragma: no branch (the full design has rank p)
        if np.linalg.matrix_rank(design[order[:k]]) == p:
            return order[:k]


def _lts_steps(design, y, coef, h, max_steps=200):
    """C-steps: refit to the h smallest squared residuals while the objective decreases."""
    objective, subset = np.inf, None
    for _ in range(max_steps):
        squared = (y - design @ coef) ** 2
        updated = np.sort(np.argpartition(squared, h - 1)[:h])
        new_coef = _least_squares(design[updated], y[updated])
        new_objective = np.sum((y[updated] - design[updated] @ new_coef) ** 2)
        if new_objective >= objective:
            break
        objective, subset, coef = new_objective, updated, new_coef
    return objective, subset, coef


def _warn(scale, converged):
    """Report an exact fit or an S refinement that did not converge."""
    if scale == 0:
        warnings.warn("exact fit: most residuals are zero; the scale is 0", RuntimeWarning, 3)
    elif not converged:
        warnings.warn("S did not converge within max_iter", RuntimeWarning, 3)


def _weighted_least_squares(design, y, weights):
    root = np.sqrt(weights)
    return _least_squares(design * root[:, None], y * root)


def _converged(old, new, tol):
    return np.linalg.norm(new - old) <= tol * max(tol, np.linalg.norm(old))


def _s_scale(residuals, c, breakdown, p, zero, s0=None):
    """M-scale solving ``sum(rho(r / s)) / (n - p) = breakdown``; ``|r| <= zero`` counts as 0."""
    residuals = np.where(np.abs(residuals) <= zero, 0.0, residuals)
    n = len(residuals)
    return _mscale_bisquare(residuals, c, breakdown * (n - p) / n, s0)


def _fast_s(design, y, c, breakdown, steps, random_state):
    """FAST-S: refine every elemental start a few steps, then the best to convergence.

    Returns (scale, coef, n_iter, converged) of the best candidate.
    """
    n_trials, n_best, k_steps, max_iter = steps
    rng = as_generator(random_state)
    zero = 1e-10 * mad(y)  # robust to outliers and to a shift of y
    candidates = []
    for _ in range(n_trials):
        start = _elemental_subset(design, rng)
        coef = _least_squares(design[start], y[start])
        candidate = _s_steps(design, y, coef, c, breakdown, zero, k_steps)
        if candidate[0] == 0:  # exact fit: no smaller scale exists
            return candidate
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: candidate[0])
    refined = [
        _s_steps(design, y, coef, c, breakdown, zero, max_iter)
        for _, coef, *_ in candidates[:n_best]
    ]
    return min(refined, key=lambda candidate: candidate[0])


def _s_steps(design, y, coef, c, breakdown, zero, max_steps, tol=1e-7):
    """IRWLS steps that lower the M-scale (Salibian-Barrera & Yohai 2006)."""
    p = design.shape[1]
    scale = _s_scale(y - design @ coef, c, breakdown, p, zero)
    for step in range(1, max_steps + 1):
        if scale == 0:
            return scale, coef, step - 1, True
        weights = _bisquare_weights((y - design @ coef) / scale, c)
        new_coef = _weighted_least_squares(design, y, weights)
        scale = _s_scale(y - design @ new_coef, c, breakdown, p, zero, scale)
        if _converged(coef, new_coef, tol):
            return scale, new_coef, step, True
        coef = new_coef
    return scale, coef, max_steps, False


def _bisquare(t, c):
    """``psi(t) = t (1 - (t/c)^2)^2``, ``psi'(t)`` and the derivative of ``_rho``."""
    u = (t / c) ** 2
    inside = u <= 1
    psi = np.where(inside, t * (1 - u) ** 2, 0.0)
    dpsi = np.where(inside, (1 - u) * (1 - 5 * u), 0.0)
    return psi, dpsi, 6 * psi / c**2


def _mm_covariance(design, residuals, s_residuals, scale, tuning, breakdown):
    """Covariance of the MM coefficients (Croux, Dhaene & Hoorelbeke 2003).

    The sandwich formula of ``lmrob``'s ``.vcov.avar1``, including the
    variability of the S scale; ``tuning`` is (initial, final) bisquare constant.
    """
    c0, c = tuning
    n = len(residuals)
    t, t0 = residuals / scale, s_residuals / scale
    psi, dpsi, _ = _bisquare(t, c)
    chi, dchi = _rho(t0, c0), _bisquare(t0, c0)[2]
    inverse = scale * np.linalg.inv(design.T @ (design * dpsi[:, None]))
    a = inverse @ (design.T @ (dpsi * t)) / np.mean(dchi * t0)
    b = inverse @ (design.T @ (psi * chi))
    sandwich = inverse @ (design.T @ (design * psi[:, None] ** 2)) @ inverse
    correction = np.outer(a, b) + np.outer(b, a) - np.mean(chi**2 - breakdown**2) * np.outer(a, a)
    return _positive_semidefinite(sandwich - correction / n)


def _positive_semidefinite(cov):
    """Set negative eigenvalues to 0 and keep the variances, as ``lmrob`` ("posdefify")."""
    values, vectors = np.linalg.eigh(cov)
    if values.min() >= 0:
        return cov
    fixed = (vectors * np.maximum(values, 0)) @ vectors.T
    factor = np.sqrt(np.maximum(np.diag(cov), 0) / np.diag(fixed))
    return factor[:, None] * fixed * factor


def _mm_steps(design, y, coef, scale, c, max_steps, tol=1e-7):
    """IRWLS for the bisquare M-estimator with the scale fixed."""
    for step in range(1, max_steps + 1):
        weights = _bisquare_weights((y - design @ coef) / scale, c)
        new_coef = _weighted_least_squares(design, y, weights)
        if _converged(coef, new_coef, tol):
            return new_coef, step, True
        coef = new_coef
    return coef, max_steps, False


# Pison, Van Aelst & Willems (2002), LTS scale: curves f(n) = 1 - exp(a) / n**b,
# with the constants of robustbase's LTScnp2 and LTScnp2.rew.
# (reweighted, intercept, alpha) -> [(a, b) for p = 1, two (a, b) points for p > 1]
_PISON_LTS = {
    (False, True, 0.5): [
        (0.630869217886906, 0.650789250442946),
        (-0.746945886714663, 0.56264937192689),
        (-0.535478048924724, 0.543323462033445),
    ],
    (False, True, 0.875): [
        (0.565065391014791, 1.03044199012509),
        (-0.458580153984614, 1.12236071104403),
        (-0.267178168108996, 1.1022478781154),
    ],
    (False, False, 0.5): [
        (-0.0181777452315321, 0.697629772271099),
        (-0.487338281979106, 0.405511279418594),
        (-0.340762058011, 0.37972360544988),
    ],
    (False, False, 0.875): [
        (-0.310122738776431, 1.06241615923172),
        (-0.251778730491252, 0.883966931611758),
        (-0.146660023184295, 0.86292940340761),
    ],
    (True, True, 0.5): [
        (1.58609654199605, 1.46340162526468),
        (-0.773365715932083, 2.02013996406346),
        (-0.337571678986723, 2.02037467454833),
    ],
    (True, True, 0.875): [
        (0.391653958727332, 1.03167487483316),
        (-0.474174840843602, 1.39681715704956),
        (-0.276640353112907, 1.42543242287677),
    ],
    (True, False, 0.5): [
        (0.6329852387657, 1.40361879788014),
        (-0.417574780492848, 1.83958876341367),
        (-0.175753709374146, 1.8313809497999),
    ],
    (True, False, 0.875): [
        (-0.642240988645469, 0.926325452943084),
        (-0.267522855927958, 1.17559984533974),
        (-0.161200683014406, 1.21675019853961),
    ],
}


def _lts_curve(fit, p, n):
    if p == 1:
        a, b = fit[0]
    else:  # line through log(-a_j / p^b_j) at log(3 p^2) and log(5 p^2)
        (a1, b1), (a2, b2) = fit[1], fit[2]
        y1, y2 = np.log(-a1 / p**b1), np.log(-a2 / p**b2)
        b = (y1 - y2) / np.log(5 / 3)
        a = y1 + b * np.log(3 * p * p)
    return 1 - np.exp(a) / n**b


def _lts_small_sample(p, n, alpha, intercept, reweighted):
    """Factor for the LTS scale with ``p`` slopes; interpolated in ``alpha`` as for the MCD."""
    f50 = _lts_curve(_PISON_LTS[reweighted, intercept, 0.5], p, n)
    f875 = _lts_curve(_PISON_LTS[reweighted, intercept, 0.875], p, n)
    if alpha <= 0.875:
        f = f50 + (f875 - f50) * (alpha - 0.5) / 0.375
    else:
        f = f875 + (1 - f875) * (alpha - 0.875) / 0.125
    return float(1 / f) if f > 0 else 1.0
