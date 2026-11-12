"""Robust linear regression estimators and their fitting steps.

The estimators follow the scikit-learn API: ``fit(X, y)`` takes observations in
rows and sets ``coef_``, ``intercept_`` and ``scale_`` (robust scale of the
residuals); ``predict(X)`` returns ``X @ coef_ + intercept_``.
"""

import warnings

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_generator, check_count
from .covariance import h_alpha_n, mcd_consistency

__all__ = ["LTSRegressor"]

# Residuals with |r / scale| above this get weight 0 (two-sided 2.5% per tail).
_CUTOFF = stats.norm.ppf(0.9875)


class LTSRegressor(RegressorMixin, BaseEstimator):
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
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True, ensure_min_samples=2)
        check_count("n_trials", self.n_trials)
        design = _design(X, self.fit_intercept)
        n, p = design.shape
        if n <= p:
            raise ValueError(f"LTS needs more observations than coefficients (got n={n}, p={p})")
        h = h_alpha_n(self.alpha, n, p)
        if np.linalg.matrix_rank(design) < p:
            raise ValueError("X is singular: its columns (and the intercept) are collinear")
        if h == n:
            subset = np.arange(n)
        else:
            subset = _fast_lts(design, y, h, self.n_trials, as_generator(self.random_state))
        self.h_ = h
        coef = self._raw_fit(design, y, subset)
        coef, self.scale_, self.support_ = self._reweighted_fit(design, y, coef)
        self.intercept_, self.coef_ = _split(coef, self.fit_intercept)
        return self

    def predict(self, X):
        """Predicted values ``X @ coef_ + intercept_``."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_

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
