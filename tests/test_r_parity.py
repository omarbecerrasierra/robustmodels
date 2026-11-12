"""Numerical comparisons with robustbase and rrcov.

Reference values: tests/data/r_reference.json; its "meta" entry records the R,
robustbase and rrcov versions that produced them.
Known differences: R rounds the consistency constants (e.g. 2.21914 for Qn),
uses tolerance 1e-6 in huberM and an approximate medcouple.
The outlyingness references use mrfDepth with all directions (ndir = "all"),
and robustbase's adjOutlyingness with type-5 quartiles. mrfDepth's adjusted
outlyingness differs from the exact value when rounding breaks ties among
projections (see test_outliers.py), so only its univariate values and its
flags are compared.
The LTS references use ltsReg with exhaustive search (nsamp = "exact"),
except hbk (500 random starts). S and MM references come from lmrob, which
rounds its bisquare constants (1.54764, 4.685061): scales differ by about 3e-6.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from robustmodels import location, scale
from robustmodels.covariance import MCD, MRCD, OGK, MMEstimator, SEstimator, mcd_consistency
from robustmodels.outliers import AdjustedOutlyingness, StahelDonohoOutlyingness
from robustmodels.plots import _adjusted_box_stats
from robustmodels.regression import LTSRegressor, MMRegressor, SRegressor, _lts_small_sample

R = json.loads((Path(__file__).parent / "data" / "r_reference.json").read_text())
UNI = R["univariate"]
MV = R["multivariate"]
REG = R["regression"]
LTS_DATA = [name for name in REG if name != "small_sample"]

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


@pytest.mark.parametrize("name", LTS_DATA)
def test_lts_regression(name):
    r = REG[name]
    f = LTSRegressor(random_state=0).fit(np.array(r["X"]), np.array(r["y"]))
    np.testing.assert_array_equal(np.flatnonzero(f.raw_support_), np.sort(r["best"]))
    assert close(np.r_[f.raw_intercept_, f.raw_coef_], r["raw_coef"], 1e-12)
    assert f.raw_scale_ == pytest.approx(r["raw_scale"], rel=1e-12)
    assert close(np.r_[f.intercept_, f.coef_], r["coef"], 1e-12)
    assert f.scale_ == pytest.approx(r["scale"], rel=1e-12)


def test_lts_small_sample_factors():
    for row in REG["small_sample"]:
        args = int(row["p"]), int(row["n"]), row["alpha"], row["intercept"]
        assert _lts_small_sample(*args, False) == pytest.approx(row["raw"], rel=1e-12)
        assert _lts_small_sample(*args, True) == pytest.approx(row["rew"], rel=1e-12)


@pytest.mark.parametrize("name", LTS_DATA)
def test_s_and_mm_regression(name):
    r = REG[name]["lmrob"]
    X, y = np.array(REG[name]["X"]), np.array(REG[name]["y"])
    s = SRegressor(random_state=0).fit(X, y)
    assert s.scale_ <= r["s_scale"] * (1 + 1e-5)  # S minimizes the scale: never worse than R
    if s.scale_ > r["s_scale"] * (1 - 1e-5):  # same optimum found
        assert close(np.r_[s.intercept_, s.coef_], r["s_coef"], 1e-5)
        mm = MMRegressor(random_state=0).fit(X, y)
        assert close(np.r_[mm.intercept_, mm.coef_], r["mm_coef"], 1e-5)
        assert close(mm.cov_params_, r["cov"], 1e-4)  # .vcov.avar1, summary(lmrob)
        assert close(mm.std_errors_, r["std_errors"], 1e-4)
        assert close(mm.p_values_, r["p_values"], 1e-4)


OUTL = R["outliers"]


@pytest.mark.parametrize("name", OUTL)
def test_projection_outlyingness(name):
    r = OUTL[name]
    X = np.array(r["X"], dtype=float)
    sdo = StahelDonohoOutlyingness(n_directions=10**6).fit(X)  # all hyperplanes, as ndir = "all"
    # R's mad rounds the consistency constant to 1.4826
    assert close(sdo.outlyingness_, r["sdo"], 1e-5)
    assert sdo.cutoff_ == pytest.approx(r["sdo_cutoff"], rel=1e-5)
    np.testing.assert_array_equal(sdo.predict(X) == 1, r["sdo_inlier"])
    ao = AdjustedOutlyingness(n_directions=10**6).fit(X)
    np.testing.assert_array_equal(ao.predict(X) == 1, r["ao_inlier"])
    if X.shape[1] == 1:
        assert close(ao.outlyingness_, r["ao"], 1e-12)
        assert close(ao.outlyingness_, r["ao_robustbase"], 1e-12)


@pytest.mark.parametrize("name", R["adjbox"])
def test_adjusted_boxplot_statistics(name):
    r = R["adjbox"][name]
    s = _adjusted_box_stats(np.array(r["x"]))
    stats = [s["whislo"], s["q1"], s["med"], s["q3"], s["whishi"]]
    assert close(stats, r["stats"], 1e-12)  # robustbase::adjboxStats
    assert close(s["fences"], r["fence"], 1e-12)
    np.testing.assert_array_equal(np.sort(s["fliers"]), np.array(r["out"], dtype=float))
