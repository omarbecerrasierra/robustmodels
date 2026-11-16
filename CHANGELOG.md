# Changelog

## 0.2.0 (2026-11-16)

- Add Python 3.15 to the Linux, Windows and macOS CI matrix and compatibility metadata.
- Audit the implemented formulas against primary theory and independent numerical
  checks; document assumptions, computational variants and remaining limits.
- Correct ROBPCA's published subset size and preserve it in the final MCD;
  retain infinite outlyingness for off-hyperplane points when projection scale
  is zero, and reject unidentified or numerically degenerate components.
- Reject undefined Hotelling moment calibrations and singular MANOVA statistics;
  preserve every original group through trimming.
- Correct MT transformation minima near zero and differentiate its clipped
  numerical objective; capped fits report non-convergence and unavailable inference.
  Exclude `c<=1`; larger constants still require the documented regularity assumptions.
- Reject rank-deficient nonlinear model Jacobians. Document the independent,
  symmetric, homoscedastic error assumptions of nonlinear covariance.
- Correct univariate depth boundaries, narrow bivariate angular cells and
  bag interpolation with nonzero inner segments. Prevent representable
  Hodges–Lehmann/Pn results from overflowing during pairwise averaging.

- Add robust logistic and Poisson GLMs: Mallows, Bianco–Yohai (with optional
  leverage trimming) and MT, with sandwich inference and coefficient tables.
- Add nonlinear Huber M regression, final-residual scale and covariance,
  model-output validation and explicit convergence diagnostics.
- Add halfspace depth, regression depth, the bivariate Tukey median,
  bagplot statistics and `plots.bagplot`, including degenerate bags.
- Add directional outlyingness with side-specific robust scales.
- Add independent definition checks, scikit-learn checks and method guides;
  preserve frozen software outputs as supplementary regression fixtures.

- Add `multivariate`: robust principal components (`ROBPCA` and
  `ProjectionPursuitPCA`), robust discriminant analysis (`RobustLDA` and
  `RobustQDA`), the robust one-sample Hotelling test `hotelling_t2` and the
  robust one-way `manova`, with model assumptions and approximation limits.
- Add `covariance.StahelDonoho`, the Stahel-Donoho estimator of location and
  scatter with the weights of Maronna and Yohai (1995).
- Add `plots.pca_outlier_map`, `plots.scree_plot` and `plots.biplot` for robust
  PCA, `plots.discriminant_plot` for classification regions,
  `plots.distance_qq` (chi-square quantile plot of robust distances),
  `plots.residual_plot` (residuals against fitted values) and
  `plots.residual_qq` (normal quantile plot of regression residuals).
- Fix `plots.outlier_map` accepting a response whose length differs from the
  number of rows of `X`.
- `plots.distance_distance`, `plots.tolerance_ellipse` and `plots.outlier_map`
  raise `ValueError` for a singular robust covariance instead of drawing a
  97.5% reference that does not apply.
- Fix `scale.medcouple`, `lmc` and `rmc` returning `NaN` for finite values close
  to the largest float.
- `covariance.h_alpha_n` raises `ValueError` when `n <= p` instead of returning
  a subset larger than the sample.

## 0.1.3 (2026-11-12)

- Add `outliers.StahelDonohoOutlyingness` and `outliers.AdjustedOutlyingness`,
  scikit-learn outlier detectors with projection depth scores and explicit
  projection-tie handling.
- Add `regression.LTSRegressor`: least trimmed squares regression with FAST-LTS,
  reweighting and small-sample corrections.
- Add `regression.SRegressor` (FAST-S) and `regression.MMRegressor`, with
  documented tuning and scale equations.
- `MMRegressor` reports heteroskedasticity-robust standard errors, t values and
  p-values (Croux, Dhaene & Hoorelbeke 2003) and an R-style `summary()`.
- Add the Zenodo DOI to the README and `CITATION.cff`.
- Add `plots`: regression outlier maps, distance-distance plots, tolerance
  ellipses, adjusted boxplots and outlyingness plots, with Matplotlib axes
  support, class counts, infinite-score markers and separated labels.
- Use sample covariance (`n−1`) for classical plotting distances and ellipses.
- Add an executable diagnostic gallery, API reference and pre-release
  validation notes covering definitions and documented differences from R.
- Clarify that projection depth is zero for infinite outlyingness.
- Matplotlib is now a dependency.
- Show total downloads in the README badge; pypistats reads 0 for the monthly
  count during a package's first month.

## 0.1.2 (2026-11-12)

- Add a roadmap of planned methods.
- Add a page that maps robustbase, rrcov and base R functions to robustmodels.
- Give documentation pages descriptive titles, a sitemap and search descriptions.
- Credit the author by full name and ORCID, and expand the PyPI keywords.

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
