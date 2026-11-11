# Changelog

## 0.1.1 (2026-11-11)

- Stabilize univariate MCD window variances in the presence of extreme outliers.
- Finish slow M-scale iterations with a bracketed root solver.
- Select MRCD's common regularization weight according to the published rule.
- Reject MM tuning combinations that violate the loss-ordering condition.
- Warn when M-location reaches its iteration limit without convergence.
- Add method citations and clarify finite-sample, complexity and optimization limits.

- Redesign the PyPI overview with worked examples and an estimator selection guide.
- Move development and release instructions to `CONTRIBUTING.md`.
- Test every line and branch, require 100% coverage in CI, and report it to Codecov.
- Add coverage and download badges.
- Link Tarr's thesis through its permanent University of Sydney handle.
- Add `CITATION.cff`.

## 0.1.0 (2026-11-09)

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
