# Validation and numerical conventions

This page records the review performed before version 0.1.3, on
2026-11-12. It covers the mathematical definitions and numerical procedures
in location, scale, covariance, regression, projection outlyingness and the
five diagnostic plots of that release. Comparisons with R complement independent checks of
formulas and invariances; agreement with R alone is not the correctness
criterion.

The checks are evidence for the cases and assumptions described here, not a
proof of correctness for every floating-point input or a global-optimum
certificate for randomized searches. Existing documented conventions remain
part of the API.

## Definitions and procedures reviewed

| Area | Defining calculation | Independent checks |
| --- | --- | --- |
| Location | Walsh-average median for Hodges–Lehmann; fixed-scale Huber/bisquare score equations; integer-count trimming and winsorization; one-step tau location | Explicit pair enumeration, score residuals, shift/scale equivariance, contamination and zero-scale cases |
| Scale and skewness | Qn order statistic, Sn high/low medians, Pn inverse empirical-CDF quartiles, MAD, truncated-square tau scale, bisquare M-scale, medcouple tie kernel | Brute-force definitions, Gaussian calibration, numerical integration, finite-sample simulations, reflection and root-equation residuals |
| MCD and MRCD | Decreasing subset determinant; normal-consistency correction; common regularization weight held fixed during MRCD concentration | Exhaustive small-sample MCD comparisons, covariance transformations, positive definiteness, regularization eigenvalue bound and the published common-rho rule |
| MVE and OGK | Covering-ellipsoid volume heuristic; robust pairwise covariance followed by orthogonalization | Covered subsets, transformation properties, contamination behavior and normal-model calibration; no claim of exact MVE optimization |
| Multivariate S/MM | Unit-determinant shape, bisquare scale equation, fixed S-scale in MM, ordered losses | Scale-equation residuals, decreasing loss, determinant preservation, affine transformations and Gaussian shape-efficiency constants |
| LTS and regression S/MM | Trimmed residual sum of squares; corrected residual-scale equation; fixed-scale MM refinement | Exhaustive small LTS problems, regression/scale equivariance, estimating equations, loss decrease and simulations |
| MM inference | Sandwich covariance including uncertainty in the initial S-scale | Formula comparison with R using identical inputs, symmetry/positive-semidefinite handling, Gaussian efficiency and heteroskedastic interval simulations |
| Projection outlyingness | Largest standardized projection over selected hyperplanes; side-specific adjusted whiskers; depth and fitted log cutoff | Affine transforms, convexity, exhaustive low-dimensional directions, exact integer-data projections and zero-scale behavior |
| Diagnostics | Unsquared distances, standardized residuals, reference contours, adjusted fences and the detector's own flags | Coordinates and class counts, ellipse quadratic identities, observed whisker endpoints, reflection, infinite-score display and label separation |

The linked [method references](references.md) identify the sources and
conventions for each public function. The existing automated suite includes
brute-force and simulation checks; the additional R runs and numerical audit
were executed separately, without adding new R-dependent tests or fixtures.

### Independent numerical calculations

Additional quadrature evaluated the defining expectations directly, rather
than reusing the implementation's closed-form chi-square moments:

| Check | Cases | Largest absolute discrepancy |
| --- | --- | --- |
| `E[rho_c(||Z||)] = breakdown` | Dimensions 1, 2, 5, 10; breakdown 0.1, 0.3, 0.5 | `7.8e-16` |
| Bisquare regression efficiency `E[psi']² / E[psi²]` | Efficiencies 0.5, 0.85, 0.95 | `1.9e-15` |
| Bisquare shape-efficiency expectation | Those efficiencies and dimensions 1, 2, 5, 10 | `9.1e-14` |
| MCD consistency times truncated normal variance | Dimensions 1, 2, 5, 10; coverages 0.5, 0.75, 0.975 | `5.6e-16` |
| Ellipse quadratic form minus its chi-square threshold | 20 rotated positive-definite covariance matrices | `8.9e-15` |
| Medcouple versus explicit kernel median | Sample sizes 8, 9, 40, 41, 200 | `0` |

On three additional contaminated regression samples, the largest normalized
MM score residual was `8.6e-9`; the largest S scale-equation residual was
`8.4e-12`. Every MM refinement reduced its final bisquare loss from the S
starting fit. These checks assess the equations and descent, not global
minimality.

## Direct R comparison

The separate comparison used **R 4.6.2**, **robustbase 0.99-7**,
**rrcov 1.7-7** and **mrfDepth 1.0.17**. It evaluated the same input arrays in
both languages, including `hbk`, `stackloss`, `starsCYG` and `wood`, plus
symmetric, skewed, reflected, tied, constant and singleton samples for the
adjusted boxplot. A seed does not produce identical random subsets across
languages; deterministic fits or aligned definitions were used where possible.

| Comparison | Observed result |
| --- | --- |
| LTS versus `ltsReg` on four datasets | Identical raw subsets; coefficient and scale discrepancies below `5e-15` after division by `max(1, max(abs(R result)))` |
| Deterministic MCD versus `covMcd` and `CovMcd` on hbk, starsCYG and wood | Covariances agree within `7e-16` absolute; robust distances within `3e-14`; identical flags |
| Classical distances versus `mahalanobis(X, colMeans(X), cov(X))` | Largest absolute discrepancy `1.1e-13` after using divisor `n−1` |
| OGK versus `covOGK` on four datasets | Raw covariance discrepancies below `3.3e-13` on the normalized metric above; reweighted covariance agrees after accounting for the documented consistency factor |
| MRCD versus `CovMrcd` on four datasets | Covariance discrepancies below `7.4e-7` on that normalized metric; R's small positive regularization floor differs from an exact zero weight |
| Adjusted boxplot versus `adjboxStats` on 34 samples | Identical hinges, medians, observed whiskers and flagged values; some fences differ by up to `0.00154` because R approximates the medcouple |
| Regression-map classes with MM fits on four datasets | Identical classifications with matched predictor-distance degrees of freedom; hbk has 61 regular, 4 good-leverage and 10 bad-leverage points |
| MM covariance formula with identical residuals, scale and tuning | Largest absolute discrepancy below `3.5e-12` against `.vcov.avar1` |
| SDO and adjusted-outlyingness flags on univariate, starsCYG and stackloss data | Identical flags for both detectors; values need not coincide under the documented MAD and tie conventions |

For the regression-map comparison, R's `lmrob` used exact Python tuning
constants, five refined S candidates and 2,000 starts. Python used its default
500 starts. The finite searches may select different S solutions. In hbk,
initial S residuals differ although the S scales and final MM coefficients
agree closely; the covariance formula uses those initial residuals, so the
resulting standard errors need not be identical. Comparing the covariance
formula with the **same** inputs separates this optimization difference from
a formula error.

For multivariate S on wood, Python found a scale of `0.03992495` versus
`0.04065488` in the R run. A smaller scale is a better value of the S
objective; it does not prove that either search found the global optimum.
The resulting S and MM covariance matrices are therefore not required to
match entry by entry. On the other reviewed datasets, the scales agree and
matrix differences are consistent with iteration tolerances.

## Decisions retained

- **Exact versus rounded calibration.** MAD, Qn, Sn and bisquare constants
  follow the documented definitions. Rounded R constants and stopping
  tolerances explain small discrepancies in the univariate comparisons.
- **Medcouple ties and approximation.** The kernel includes ties at the
  median and averages the middle pair when there are an even number of
  kernels. This can differ from robustbase, and mrfDepth's projection ties
  can also change adjusted outlyingness. Existing exact-arithmetic checks
  on stackloss validate the chosen definition independently of R.
- **Quartiles.** Pn uses inverse empirical-CDF quartiles; adjusted boxplots
  use Tukey hinges; adjusted outlyingness uses type-5 projection quartiles.
  These are deliberate, separate conventions.
- **OGK correction.** The reweighted covariance includes a normal-consistency
  factor omitted by R's `covOGK`. Comparisons account for it rather than
  removing it from the implementation.
- **Regression scale.** S regression divides its scale equation by `n−p`,
  including the intercept in `p`. Reweighted LTS uses its documented
  `sum(weights)−1` denominator and empirical small-sample factors. These are
  calibration conventions, not claims of exact unbiasedness under contamination.
- **MRCD regularization.** The published common-rho rule is retained; its
  condition target applies to initial subsets in standardized coordinates,
  not as a universal bound on the final covariance in original units.
- **Efficiency and breakdown.** Multivariate MM tunes shape efficiency;
  regression MM tunes coefficient efficiency. The starting S-scale is held
  fixed and final tuning must preserve the ordering of normalized losses.
  The usual percentages are asymptotic statements under model assumptions,
  not guarantees for every sample size and dimension.
- **Searches and convergence.** Random starts and local refinements can find
  different solutions. The MVE search is a covering-ellipsoid heuristic;
  exact parity with a different heuristic is not an appropriate acceptance
  criterion. Finite searches do not certify global optima.

## Plot corrections and interpretation

The review changed classical plotting covariances from divisor `n` to `n−1`,
matching the documented sample covariance and R's classical distances.
The theoretical checks verify the plotted ellipse boundary directly against
the covariance used for flagging. The extra `n / (n−1)` inflation in
robustbase 0.99-7's ellipse drawing routine is intentionally not reproduced.

The regression distance cutoff uses the number of predictor columns, without
an intercept column. This matches the dimension of the MCD distance; it can
differ from `plot.lmrob`'s use of regression rank. Rows beyond both cutoffs
are bad leverage points, not merely observations with high leverage.

The chi-square contour is a normal-model reference. It is not a confidence
region for the center or an exact finite-sample false-positive guarantee.
Degenerate covariance matrices invalidate the usual chi-square calibration;
a pseudoinverse alone does not resolve that issue, so the distance-distance
plot, the tolerance ellipses and the regression outlier map raise `ValueError`
when the robust covariance is singular. The regression outlier map also
requires one response per row of `X`. Outlyingness can be
infinite when a projection scale is zero, in which case projection depth is
zero. The graphic distinguishes that mathematical value from its finite
display position.

## Changes after the independent review of 0.1.3

An external review of 0.1.3 reproduced four defects, all corrected:

- `plots.outlier_map` accepted a response of the wrong length, which NumPy
  broadcast silently.
- `plots.distance_distance` and `plots.tolerance_ellipse` drew 97.5% references
  for a singular covariance.
- `scale.medcouple` returned `NaN` for finite values near the largest float,
  because differences overflowed; the sample is now rescaled by an exact
  power of two, which leaves other results unchanged.
- `covariance.h_alpha_n` returned a subset larger than the sample for `n <= p`.

Conventions the review asked about and that are kept: trimmed and winsorized
means remove `floor(n * proportion)` observations per tail, as R's `mean`, so
small samples can be left untrimmed; outlyingness is infinite when a projection
has zero MAD; and an exact regression fit gives zero scale and undefined
standard errors.

## Multivariate methods

ROBPCA, the discriminant rules, Hotelling's test and Wilks' lambda were
compared with rrcov 1.7-7 on the reference data in
`tests/data/r_multivariate.json`. They agree to rounding error wherever both
programs evaluate the same deterministic quantity; the remaining differences
are documented choices (the orthogonal cutoff of ROBPCA, the weight function of
the Stahel-Donoho estimator and the MCD search used by MANOVA). The level of
the two tests was checked by simulation under the normal model: between
3.7% and 6.7% of rejections at the 5% level in the settings tried. See the
[multivariate guide](multivariate.md#numerical-limits-and-sources).

## Repeating the maintained checks

```bash
uv run pytest --cov
uv run pytest --doctest-modules --pyargs robustmodels
uv run ruff check .
uv run ruff format --check .
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

The documentation build executes the plotting examples and renders their
figures. The maintained tests check the coordinates, mathematical invariants,
input handling and plotting interface. Coverage measures exercised code;
it does not replace statistical validation. The separate R comparisons above
are a record of this review and do not add an R runtime dependency to CI.


## Additional methods (unreleased)

`tests/test_glm.py`, `tests/test_nonlinear.py` and `tests/test_depth.py`
add frozen robustbase/mrfDepth witnesses, direct summation of Mallows moments,
finite differences of BY and MT derivatives, an analytic nonlinear Jacobian,
exact small halfplane enumeration, exhaustive regression separation and
bivariate transformation checks. Directional scales are independently checked
by Gaussian quadrature in `tests/test_outliers.py`. Plot tests compare the
bagplot polygons and points with its underlying statistics.

The frozen JSON files in `tests/data` record the R reference results and
package versions. Running the Python tests requires no R installation. The [GLM](glm.md), [nonlinear regression](regression.md) and
[depth](depth.md) guides describe intentional R differences and invalid
reference values that are not used as correctness targets.
