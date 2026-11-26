"""Checks for finite samples and estimator parameters."""

from numbers import Real

import numpy as np


def as_1d(x):
    """``x`` as a finite 1-D float array."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or infinite values")
    return x


def check_range(name, value, low, high, low_open=False, high_open=False):
    """Raise ``ValueError`` unless ``value`` lies in the interval ``[low, high]``."""
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    below = value <= low if low_open else value < low
    above = value >= high if high_open else value > high
    if not np.isfinite(value) or below or above:
        interval = f"{'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}"
        raise ValueError(f"{name} must be in {interval}, got {value}")


def check_count(name, value):
    """Require a positive integer for a trial or iteration count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def as_generator(random_state):
    """NumPy ``Generator`` from None, an int, a Generator or a legacy RandomState."""
    if isinstance(random_state, np.random.RandomState):
        random_state = random_state.randint(2**31)
    return np.random.default_rng(random_state)


def check_option(name, value, options):
    """Raise ``ValueError`` unless ``value`` is one of ``options``."""
    if value not in options:
        raise ValueError(f"{name} must be one of {list(options)}, got {value!r}")


def is_exact_fit(y, residuals):
    """Whether all residuals are within roundoff at the scale of the response."""
    return np.max(np.abs(residuals)) <= 32 * np.finfo(float).eps * np.max(np.abs(y))


def unit_exponent(x, reference=0.0, *, axis=None):
    """Power-of-two rescaling that preserves small observations in a wide range."""
    magnitude = np.abs(x)
    largest = np.maximum(magnitude.max(axis=axis), abs(reference))
    smallest = magnitude.min(axis=axis, where=magnitude > 0, initial=np.inf)
    if reference != 0:
        smallest = np.minimum(smallest, abs(reference))
    exponent = np.frexp(largest)[1]
    # Leave the smallest normal observation in the normal range.
    normal_limit = np.frexp(smallest)[1] - np.finfo(float).minexp - 1
    return np.minimum(exponent, np.maximum(0, normal_limit))


def column_scaling(X, center=True):
    """Median (or 0 without ``center``) and power-of-two spread of each column of ``X``."""
    location = np.median(X, axis=0) if center else np.zeros(X.shape[1])
    deviation = np.abs(X - location)
    spread = np.median(deviation, axis=0)
    spread = np.where(spread > 0, spread, deviation.max(axis=0))
    return location, np.exp2(np.round(np.log2(np.where(spread > 0, spread, 1.0))))


def standardize_columns(X, center=True):
    """Columns of ``X`` centred and scaled to unit spread, and the map back to ``X``.

    Returns ``(Z, back)``. A fit ``[a, b]`` on ``[1, Z]`` (or ``b`` on ``Z`` without
    ``center``) is ``back @ [a, b]`` on ``[1, X]`` (or ``X``), and a covariance ``V``
    is ``back @ V @ back.T``. Affine equivariant fits are computed on ``Z`` because
    large means or very different column scales make ``X`` ill-conditioned. Scales
    are powers of two, so rescaling alone is exact.
    """
    location, spread = column_scaling(X, center)
    back = np.diag(1 / spread)
    if center:
        corner = np.zeros((len(spread), 1))
        back = np.block([[np.ones((1, 1)), -(location / spread)[None]], [corner, back]])
    return (X - location) / spread, back
