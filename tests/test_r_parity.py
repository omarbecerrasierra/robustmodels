"""Numerical comparisons with robustbase and rrcov.

Reference values: tests/data/r_reference.json; its "meta" entry records the R,
robustbase and rrcov versions that produced them.
Known differences: R rounds the consistency constants (e.g. 2.21914 for Qn),
uses tolerance 1e-6 in huberM and an approximate medcouple.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from robustmodels import location, scale
from robustmodels.covariance import MCD, MRCD, OGK, MMEstimator, SEstimator, mcd_consistency

R = json.loads((Path(__file__).parent / "data" / "r_reference.json").read_text())
UNI = R["univariate"]
MV = R["multivariate"]

UNIVARIATE = [
    ("Qn", scale.qn, 1e-5),
    ("Sn", scale.sn, 1e-5),
    ("mad", scale.mad, 1e-5),
    ("scaleTau2", scale.tau_scale, 1e-12),
    ("tau_location", location.tau_location, 1e-12),
    ("huberM", lambda x: location.m_location(x, c=1.5), 1e-5),
    ("mc", scale.medcouple, 1e-4),
    ("lmc", scale.lmc, 1e-4),
    ("rmc", scale.rmc, 1e-4),
]


@pytest.mark.parametrize("name", UNI)
@pytest.mark.parametrize(("key", "fn", "tol"), UNIVARIATE, ids=[u[0] for u in UNIVARIATE])
def test_univariate(name, key, fn, tol):
    r = UNI[name]
    assert fn(np.array(r["x"])) == pytest.approx(r[key], rel=tol, abs=tol)


def close(a, b, tol):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() <= tol * np.abs(b).max()


@pytest.mark.parametrize("name", MV)
def test_mcd_and_detmcd(name):
    r = MV[name]
    X = np.array(r["X"])
    fast = MCD(random_state=0).fit(X)
    assert close(fast.raw_covariance_, r["mcd"]["raw_cov"], 1e-12)
    assert close(fast.covariance_, r["mcd"]["cov"], 1e-12)
    det = MCD(algorithm="deterministic").fit(X)
    assert close(det.covariance_, r["detmcd"]["cov"], 1e-12)


@pytest.mark.parametrize("name", MV)
def test_ogk(name):
    r = MV[name]
    X = np.array(r["X"])
    fit = OGK().fit(X)
    assert close(fit.raw_covariance_, r["ogk"]["raw_cov"], 1e-10)
    factor = mcd_consistency(X.shape[1], 0.9)  # covOGK omits this consistency factor
    assert close(fit.covariance_ / factor, r["ogk"]["cov"], 1e-10)


@pytest.mark.parametrize("name", MV)
def test_s_and_mm(name):
    r = MV[name]
    X = np.array(r["X"])
    s = SEstimator(random_state=0).fit(X)
    assert s.scale_ <= r["s"]["scale"] * (1 + 1e-8)  # S minimizes the scale: never worse than R
    if s.scale_ > r["s"]["scale"] * (1 - 1e-8):  # same optimum found
        assert close(s.covariance_, r["s"]["cov"], 1e-4)
        assert close(MMEstimator(random_state=0).fit(X).covariance_, r["mm"]["cov"], 1e-3)


@pytest.mark.parametrize("name", MV)
def test_mrcd(name):
    r = MV[name]
    assert close(MRCD().fit(np.array(r["X"])).covariance_, r["mrcd"]["cov"], 1e-5)
