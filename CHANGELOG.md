# Changelog

## 0.6.1 (2026-11-26)

### Fixed

- Fit `LTSRegressor`, `SRegressor`, `MMRegressor`, `RankRegressor`,
  `ScoreMatchingRegressor` and the four robust GLMs on centred columns scaled
  to unit spread, and map coefficients and covariances back. Columns with
  large means (about 1e7) were rejected as singular or gave wrong
  coefficients and standard errors without a warning; Mallows and score
  matching already lost accuracy at means of 1e4. `mm_test` follows.
- `NonlinearMRegressor` no longer stops at the starting value when parameters
  are large (about 1e10): steps are scaled by the Jacobian, and the Jacobian
  of the standard errors uses central differences with relative steps.
- `halfspace_depth` standardizes the columns first: with very different
  scales the smallest column was taken as a null direction in three or more
  dimensions, and the depth nearly doubled.
- `plots.distance_distance` computes the classical distances from the
  correlation matrix, which keeps columns of very different scales accurate.
- The bandwidth selectors and the KDE tests sphere the data through the
  correlation matrix, so valid columns of very different units are no longer
  rejected as singular.
- The three outlyingness detectors ignore the infinite ratios of directions
  with zero scale, as robustbase's `adjOutlyingness`. On discrete data (Likert
  items), more than half the projections tie on many directions, and up to
  all observations were flagged; `StahelDonoho` then failed with a
  `LinAlgError`.
- `PenaPrieto` skips zero-MAD directions in the same way and stops trimming
  before the retained observations lie on a hyperplane; it failed with a
  `LinAlgError` on many Likert and small-integer samples.
- `MVE` refines every candidate ellipsoid once before ranking them, as rrcov.
  With 40% of outliers, ranking raw elemental ellipsoids often missed the
  clean solution that R finds.

## 0.6.0 (2026-11-20)

### Added

- Add `density.kde_permutation_test`, a two-sample test using the integrated
  squared difference of Gaussian KDEs with a common bandwidth.
- Add `density.kde_multiscale_test`, a two-sample test with joint
  permutation calibration across several Gaussian bandwidths.
- Add a known log-mean `offset` to `MallowsPoissonRegressor.fit` and
  `predict` for unequal exposures. Preserve logistic tail probabilities
  and withhold Mallows Poisson inference above the numerical predictor cap.
- Add `datasets` with five classic data sets of robust statistics (stars of
  CYG OB1, Hawkins–Bradu–Kass, wood gravity, stack loss and hemophilia), and
  a guide that recovers their documented outliers with summaries, tests and
  diagnostic plots.
- Show an executable example for every function on the diagnostic plots
  page, including the bagplot, clustering, density and coefficient plots.

### Faster

- Fit `RankRegressor` from a BFGS pilot and a linear program on the
  closest residual pairs only, with the other pairs as a linear term
  (Portnoy & Koenker 1997). A certificate that no excluded pair changes
  sign keeps the fit exact; n = 2000 takes about a second instead of minutes.
- Reuse the implicit Qn selector for the Wilcoxon scale on large samples,
  with linear auxiliary memory and the same treatment of ties. Use
  condensed SciPy distances and NumPy partitioning for small samples.

### Fixed

- Fix `rank_anova` and `drop_test` results that depended on the group labels
  or contrasts: when the rank fit is not unique, return the minimizer with
  the smallest residual sum of squares.
- Validate response alignment and design nesting in `rank.drop_test`.
  Handle one residual degree of freedom and report unavailable inference when
  the residual scale cannot be estimated.
- Accept NumPy scalar shifts in `signed_rank_test` without overflow warnings
  and reject booleans.
- Preserve paired signed-rank differences when large terms cancel or a
  partial sum overflows. Reject differences that overflow float64.
- Reject degenerate normal calibration in `kde_test` and validate fixed
  bandwidths independently of their scale. Normalize KDE derivatives without
  forming the determinant.
- Preserve log-concave density and CDF calculations at extreme scales.
  Handle small samples and exact fits in score matching regression.
- Test covariance rank on the correlation scale so changes in measurement
  units do not create spurious singular fits. Compute PCA residuals before
  restoring the original coordinate origin.
- Preserve finite univariate estimates at extreme scales. Compute halfspace
  depth within the affine span of lower-dimensional data and calibrate
  outlyingness using the finite training scores.
- Keep TCLUST assignments, weights and likelihood consistent at the
  iteration limit and report `converged_`. Fit in normalized units.
- Validate the data and tested columns in `mm_test`. Compute nonlinear
  regression and diagnostic OLS uncertainty without squaring the design's
  condition number and reject unidentified OLS reference intervals.
- Share the package's MAD between scale estimation, Stahel–Donoho
  projections and outlyingness cutoffs. Preserve finite medians when
  intermediate sums or deviations overflow.
- Scale the shared least absolute deviations solver internally so that
  changing regression units preserves the optimization problem numerically.
- Use SciPy's trimmed mean, softmax, ordinal ranks and cumulative
  trapezoidal integrals. Reuse OGK's tau location from `location` and share
  the adjusted fences, score weights and zero-scale division rules.

## 0.5.0 (2026-11-19)

- Add `density`: `KernelDensity`, kernel density estimation with a normal
  kernel and a full bandwidth matrix, with its gradient, Hessian and highest
  density levels; the plug-in and smoothed cross-validation bandwidth
  selectors `plugin_bandwidth` and `scv_bandwidth`, for the density or its
  first or second derivative; and `kde_test`, a two-sample test of equal
  densities in any dimension.
- Add `shape`: `LogConcaveDensity`, the maximum likelihood estimate of a
  univariate log-concave density, and `ScoreMatchingRegressor`, linear
  regression with a convex loss estimated by antitonic score matching, with
  standard errors and `summary()`.
- Add `plots.hdr_plot`, `plots.log_concave_plot` and `plots.score_plot`.
- Add `plots.coefficient_plot` (estimates and intervals of several fits, with
  least squares as reference), `plots.dd_plot` (depth-depth plot of two samples)
  and `plots.ctl_curves` (trimmed likelihood curves for choosing the number of
  clusters and the trimming level, as R's `ctlcurves`).
- Plot layout: the legends of `tolerance_ellipse`, `bagplot` and
  `discriminant_plot` move under the axes instead of covering the data;
  `distance_distance` leaves room for its labels; `adjusted_boxplot` shows each
  medcouple under its box name; `bagplot` labels its axes x₁ and x₂, as the
  other plots.
- `rank.RankRegressor` and the new regressor share one least absolute
  deviations solver (the dual linear program).

## 0.4.0 (2026-11-18)

- Add `cluster.TClust`: robust clustering by trimming with eigenvalue
  restrictions (TCLUST), including trimmed k-means, and `plots.cluster_plot`.
- Add `regression.mm_test`: robust deviance and Wald tests of nested MM
  regressions, as R's `anova.lmrob`.
- Add `rank.rank_anova`: one-way analysis of variance by ranks, as Rfit's
  `oneway.rfit`.

## 0.3.0 (2026-11-18)

- Add `rank`: `RankRegressor`, rank-based linear regression with Wilcoxon
  scores (Jaeckel's dispersion minimized exactly by linear programming), with
  standard errors, `summary()` and `drop_test` for nested models; and
  `rank_sum_test` and `signed_rank_test`, exact Wilcoxon tests conditional on
  ties and zeros (Wilcox or Pratt zeros).
- Add `covariance.PenaPrieto`, the Peña–Prieto kurtosis and specific directions
  (KSD) estimator of location and scatter for many variables, with Maronna's
  observation directions.
- Add `exact=True` to `depth.halfspace_depth`: exact depth in any dimension by
  the recursive algorithm of Dyckerhoff and Mozharovskyi (2016).
- Fix bivariate `depth.halfspace_depth` returning a depth one count too low
  when the query lies between two observations on a line (for example exact
  integer data): opposite offsets now share one line angle. This also affects
  the bagplot depth regions of such data.

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
