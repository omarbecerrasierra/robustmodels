"""Brute-force definitions used to check the numerical algorithms.

Deliberately O(n^2) or exhaustive, with no algorithmic shortcuts.
"""

import itertools
import math

import numpy as np


def pairs(x, op, diagonal=False):
    i, j = np.triu_indices(len(x), 0 if diagonal else 1)
    return op(x[i], x[j])


def mean2(a, b):
    return (a + b) / 2


def qn_raw(x):  # Rousseeuw & Croux (1993)
    h = len(x) // 2 + 1
    return np.sort(np.abs(pairs(x, np.subtract)))[h * (h - 1) // 2 - 1]


def sn_raw(x):  # lomed_i himed_j |x_i - x_j|
    himed = np.sort(np.abs(np.subtract.outer(x, x)), axis=1)[:, len(x) // 2]
    return np.sort(himed)[(len(x) + 1) // 2 - 1]


def pn_raw(x):  # Tarr, Müller & Weber (2012)
    w = np.sort(pairs(x, mean2))
    return w[math.ceil(3 * w.size / 4) - 1] - w[math.ceil(w.size / 4) - 1]


def hodges_lehmann(x):
    return np.median(pairs(x, mean2, diagonal=True))


def medcouple(x):  # Brys, Hubert & Struyf (2004), with the kernel for ties at the median
    x = np.sort(x)[::-1]
    m = np.median(x)
    zp, zm = x[x >= m] - m, x[x <= m] - m
    p = len(zp)
    h = [
        np.sign(p - 1 - i - j) if a == b else (a + b) / (a - b)
        for i, a in enumerate(zp)
        for j, b in enumerate(zm)
    ]
    return float(np.median(h))


def mcd_logdet(X, h):
    """Smallest log det(cov) over all h-subsets (exact MCD objective)."""
    return min(
        np.linalg.slogdet(np.cov(X[list(s)].T))[1] for s in itertools.combinations(range(len(X)), h)
    )
