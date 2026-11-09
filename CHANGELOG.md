# Changelog

## 0.1.0 (unreleased)

- Handle a zero starting MAD in M-scale and degenerate MRCD subsets explicitly.
- Check the source distribution, wheel metadata and installed package in CI.
- Add Tarr's finite-sample correction for Pn, enabled by default; use
  `finite_correction=False` for the previous result.
- Keep location, scale and covariance in three modules, with estimator steps and
  correction factors beside their implementations. Only input validation is separate.
- Validate tuning parameters and iteration counts; handle zero bisquare weights.
- Reject full coverage in MVE, where the scale correction is undefined.
- Remove the typing marker until the package provides type annotations.
- `scale`: `mad`, `qn`, `sn`, `pn`, `tau_scale`, `m_scale`, `medcouple`, `lmc`, `rmc`.
- `location`: `hodges_lehmann`, `m_location`, `tau_location`, `trimmed_mean`, `winsorized_mean`.
- `covariance`: `MCD` (FAST-MCD and DetMCD), `MVE`, `OGK`, `SEstimator`, `MMEstimator`, `MRCD`.
