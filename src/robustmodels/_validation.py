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
