# Validation and numerical conventions

Mathematical definitions, assumptions and independently calculated consequences
are the acceptance criteria for robustmodels. The theoretical audit dated
2026-11-13 found reproducible errors even though the earlier software tests
passed. The corrected counterexamples and source equations are recorded in the
[scientific audit report](https://github.com/omarbecerrasierra/robustmodels/blob/main/reports/Auditor%C3%ADa%20te%C3%B3rica%20de%20robustmodels.md).

These checks support the cases described here. They do not prove every
floating-point input correct, certify global optima of randomized searches,
or establish superiority over another implementation. Coverage measures
exercised code, not mathematical validity.

## Definitions and independent evidence

| Area | Defining calculation | Independent checks |
| --- | --- | --- |
| Location and scale | Walsh-average median; Qn/Sn order statistics; Pn pair-mean quantiles; score and scale equations; medcouple tie kernel | Explicit enumeration, analytic calibration, quadrature, shift/scale/reflection properties and representable extreme values |
| Covariance | Trimmed determinant, Gaussian truncation correction, common MRCD regularization, S scale and fixed-scale MM shape | Exhaustive small problems, direct Gaussian radial integration, positive definiteness, transformation properties and loss ordering |
| Linear regression | Trimmed sum of squares; S scale equation; fixed-scale MM criterion and sandwich | Exact small LTS problems, score residuals, descent, regression equivariance and model-based simulations |
| Nonlinear regression | Huber IRWLS with a final-residual scale and model Jacobian | Analytic exponential Jacobian, non-identifiable-parameter rejection, contamination and exact fits |
| GLMs | Corrected conditional scores, sensitivities and sandwich variance | Direct Bernoulli/Poisson expectations, finite-difference derivatives, low-mean MT correction equations and numerical-cap cases |
| PCA and classification | Published ROBPCA h, projection/orthogonal distances, normal discriminant scores | Actual subset-size checks, orthonormality, reconstruction, rotations, exact fits and direct normal densities |
| Mean-vector tests | Robust T² with finite-moment F calibration; weighted Wilks statistic | Undefined-moment/singularity guards, preservation of original groups and bounded independent null simulations |
| Depth and outlyingness | Closed halfspaces, regression separation, interpolated depth regions and directional side scales | Rational-sign halfplane enumeration, exact 1-D boundaries, exhaustive deletion with linear programming, tied segments and Gaussian quadrature; exact depth in three and four dimensions against linear-programming separation of every subset |
| High-dimensional covariance | KSD projection steps, Mahalanobis completion and chi-square consistency | Recovery of planted outliers, consistency on 4000 normal observations, affine equivariance with fixed draws, and RobStatTM's selected observations |
| Robust tests and clustering | Robust deviance and Wald statistics; trimmed classification likelihood with restricted eigenvalues | Simulated 5% size with normal and t3 errors; brute-force eigenvalue truncation, fixed-point assignment and trimmed k-means nearest centers |
| Rank methods | Jaeckel's dispersion as an L1 problem on pairwise differences; exact conditional distributions of doubled midranks | Enumeration of every assignment and sign pattern with ties and zeros, local optimality of the pairwise loss, the weighted median of pairwise slopes, equivariance, population values of `tau` and `tau_s`, and simulated coverage and test size |
| Diagnostics | Fitted scores, distances, cutoffs, polygon boundaries and class counts | Coordinates, ellipse quadratic identities, observed endpoints, infinite-score display and Matplotlib interface |

Scientific sources are linked in the [method-to-source map](references.md).
The audit notes and report distinguish published results from calculations
performed for this project and from assumptions that were not fully verified.

## Counterexamples corrected in the audit

- ROBPCA now uses the Appendix subset size and preserves it in final MCD.
  A zero robust projection scale retains infinite outlyingness for points
  away from the corresponding hyperplane; unidentified components are rejected.
- MANOVA rejects singular statistics and retains every original group through
  trimming. Hotelling rejects nonpositive or nonfinite scale calibration and
  `q<=4`, where the two F moments used by the approximation do not exist.
- MT searches the first correction-grid interval. At Poisson mean 0.05,
  direct summation of its defining score changed from about `0.0288` to
  `-5.0e-7`; the maintained test checks small means with a `3e-6` tolerance.
  Its gradient now differentiates the implemented clipped objective, and a
  final fit at the predictor cap has unavailable inference and reports
  non-convergence. `c<=1` is excluded by the near-zero transformation argument.
- Nonlinear regression rejects a rank-deficient final model Jacobian, rather
  than reporting individual-parameter standard errors from a pseudoinverse.
- Halfspace depth uses direct 1-D comparisons and open bivariate angular
  cells. A narrow seven-point example now gives the rationally enumerated
  `2/7`, rather than `3/7`. An additional 1000 narrow samples agreed with
  independent exact-sign enumeration; this is finite evidence, not a proof
  for all IEEE-754 inputs.
- Bivariate halfspace depth gives opposite offsets one line angle. With
  exact data and the query between two observations, separately rounded
  angles could open a spurious cell in 0.2.0: 8 of 3000 midpoint queries of
  small integer samples returned a depth one count too low. All 3000 now agree
  with rational-sign enumeration.
- Bag interpolation retains nonzero inner segments. The tied five-point
  example has the independently derived diagonal boundary `(13/6,13/6)`.
- Safe pair and central averages prevent intermediate overflow in
  Hodges–Lehmann and Pn when the mathematical result is representable.

The tests preserve these counterexamples as regressions of the definitions.
No numerical agreement with an external program was required to choose a fix.

## Assumptions and computational limits

**GLM and nonlinear inference.** The Mallows covariance is conditional on the
fixed design and model distribution. BY's corrected score is checked directly
under Bernoulli probabilities. MT uses a bisquare criterion and an approximate
spline correction; the unique, increasing transformation and source-theorem
regularity conditions are not proved for every accepted tuning. In particular,
the smooth-loss example in the source uses exponent 4, while ordinary bisquare
uses exponent 3. `c>1` is only a necessary guard in this implementation.
Nonlinear covariance assumes a correctly specified model with independent,
symmetric, homoscedastic errors and a full-rank Jacobian. It is not a
heteroscedasticity-robust covariance. See [GLMs](glm.md) and
[nonlinear regression](regression.md#nonlinear-robust-regression).

**PCA and robust mean tests.** ROBPCA is a documented computational variant
using generic corrected MCD in its final stage; not every Appendix fitting
and scale step is reproduced. Hotelling and MANOVA use approximate normal-model
calibrations. The independent null experiments and exact binomial confidence
intervals in the [multivariate guide](multivariate.md#numerical-limits-and-sources)
show sampling uncertainty and a liberal Hotelling case. They do not establish
uniform nominal level, power or robustness under arbitrary contamination.

**Outlyingness and depth.** Directional side-scale breakdown results require
the source's continuous symmetric unimodal model; they describe 25% implosion
and explosion breakdown of those scales. The log-scale cutoff is approximate,
not an exact false-positive guarantee. Higher-dimensional depth searches can
overestimate the exact depth, while finite outlyingness directions can
underestimate a supremum. Bag boundaries use a finite angular grid and large
samples use a subset. See [outliers](outliers.md) and [depth](depth.md).

**Covariance and regression conventions.** Finite-sample factors are empirical
calibrations. MRCD's common-rho condition target applies to initial subsets in
standardized coordinates, not every final covariance in original units.
Multivariate MM tunes shape efficiency and regression MM tunes coefficient
efficiency. Scale equations, fixed initial scale and normalized-loss ordering
must be preserved. Finite MCD/MVE/S/MM/LTS searches can select local solutions.
The audit found no demonstrated reason to change the existing covariance
formulas; it did not certify every global optimum or breakdown theorem.

**Plots.** Normal-reference chi-square contours describe model distances,
not confidence regions for a center or exact finite-sample error rates.
Singular covariances invalidate that reference and raise `ValueError` in the
relevant distance/ellipse plots. Plotting an infinite outlyingness at a finite
display position does not replace its mathematical value.

## Software references and reproducibility

Frozen JSON fixtures in `tests/data` record outputs and package versions from
R's robustbase, rrcov, mrfDepth, aplpack, ddalpha, RobStatTM, Rfit,
exactRankTests, coin and tclust. Their provenance remains visible.
They are supplementary software regression evidence and migration aids, not
mathematical oracles. A difference is evaluated against the selected definition,
assumptions and numerical convention. The [R migration guide](from-r.md)
records corresponding interfaces and useful differences. Python tests require
no R installation, and no R scripts are included.

The CI matrix covers Python 3.10–3.15 on Linux, Windows and macOS. Python 3.10
also checks the minimum supported numerical dependencies. Optional pandas or
array-API checks can be skipped when those optional dependencies are absent.
Expected generic estimator-check failures are documented with their mathematical
reason, and equivalent valid-data properties are tested where applicable.

```bash
uv run pytest --cov
uv run pytest --doctest-modules --pyargs robustmodels
uv run ruff check .
uv run ruff format --check .
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

The documentation build executes plotting examples. The final suite counts and
cross-platform status belong to the run for the reviewed commit; this page does
not reuse counts from an earlier implementation as evidence for later changes.
