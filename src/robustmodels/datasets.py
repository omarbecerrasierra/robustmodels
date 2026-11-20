"""Classic data sets of robust statistics.

Each loader returns a :class:`Dataset` with the predictors ``X``, the response
or class labels ``y`` and the column names. The values are those distributed
with R: ``starsCYG``, ``hbk`` and ``wood`` from robustbase 0.99-7,
``stackloss`` from base R and ``hemophilia`` from rrcov 1.7-7. The outliers
named in each docstring are the ones documented in the literature; row
numbers are zero-based.
"""

from importlib.resources import files
from typing import NamedTuple

import numpy as np

__all__ = ["Dataset", "load_hbk", "load_hemophilia", "load_stackloss", "load_stars", "load_wood"]


class Dataset(NamedTuple):
    """Predictors, response and column names of a data set."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple
    target_name: str


def load_stars():
    """Hertzsprung–Russell diagram of the star cluster CYG OB1 (47 stars).

    ``X`` is the logarithm of the effective surface temperature and ``y`` the
    logarithm of the light intensity. Rows 10, 19, 29 and 33 are red giants:
    leverage points that turn the least squares slope negative, while the
    main sequence rises (Rousseeuw & Leroy 1987, chapter 2).

    Returns
    -------
    Dataset
    """
    return _load("stars")


def load_hbk():
    """Artificial data of Hawkins, Bradu & Kass (1984): 75 rows, 3 predictors.

    Rows 0 to 13 are leverage points. Rows 0 to 9 are bad leverage points,
    off the regression of the other rows, and rows 10 to 13 are good leverage
    points, on it. Least squares and the classical covariance mask them.

    Returns
    -------
    Dataset
    """
    return _load("hbk")


def load_wood():
    """Wood specific gravity: 20 samples, 5 anatomical measurements.

    Rousseeuw (1984) replaced rows 3, 5, 7 and 18 of the data of Draper &
    Smith (1966) by outliers that are not visible in any single variable.

    Returns
    -------
    Dataset
    """
    return _load("wood")


def load_stackloss():
    """Brownlee's (1965) stack loss of a plant oxidizing ammonia: 21 days.

    ``X`` holds air flow, cooling water temperature and acid concentration
    and ``y`` the stack loss. Rows 0, 2, 3 and 20 are the outliers found by
    robust regression (Rousseeuw & Leroy 1987, chapter 3).

    Returns
    -------
    Dataset
    """
    return _load("stackloss")


def load_hemophilia():
    """Hemophilia A carriers of Habbema, Hermans & van den Broek (1974): 75 women.

    ``X`` holds the logarithms of AHF activity and AHF antigen, and ``y`` the
    group, ``"normal"`` (30) or ``"carrier"`` (45).

    Returns
    -------
    Dataset
    """
    return _load("hemophilia", target=str)


def _load(name, target=float):
    with (files(__package__) / "data" / f"{name}.csv").open() as source:
        header = source.readline().strip().split(",")
        rows = np.loadtxt(source, delimiter=",", dtype=str, ndmin=2)
    X, y = rows[:, :-1].astype(float), rows[:, -1].astype(target)
    return Dataset(X, y, tuple(header[:-1]), header[-1])
