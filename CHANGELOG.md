# Changelog

## 0.1.0 (unreleased)

First release.

- `scale`: `mad`, `qn`, `sn`, `pn`, `tau_scale`, `m_scale`, `medcouple`, `lmc`, `rmc`.
  Qn, Sn and Pn apply finite-sample corrections by default (Pn: Tarr 2014,
  Appendix B.3); pass `finite_correction=False` to disable them.
- `location`: `hodges_lehmann`, `m_location`, `tau_location`, `trimmed_mean`, `winsorized_mean`.
- `covariance`: `MCD` (FAST-MCD and DetMCD), `MRCD`, `MVE`, `OGK`, `SEstimator`,
  `MMEstimator`, with the scikit-learn estimator interface.
- Input and tuning parameters are validated; invalid values raise `ValueError`
  naming the parameter.
- Documentation at https://omarbecerrasierra.github.io/robustmodels/, with a user
  guide and a reference for every function and class; its examples run as tests.
- CI tests the installed wheel on Python 3.10 to 3.14 and publishes
  tagged releases to PyPI with trusted publishing.
