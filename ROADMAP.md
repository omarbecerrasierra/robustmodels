# Roadmap

robustmodels grows in stages. Each stage adds established methods from the
robust and nonparametric statistics literature that are missing or incomplete
in Python. Stages are ordered by how much each method is missing in Python, how
widely its R counterpart is used, and which methods later stages build on.

This is a plan, not a schedule: there are no dates, and later stages may change.
Suggestions and use cases are welcome in
[Issues](https://github.com/omarbecerrasierra/robustmodels/issues).

## Principles

- **scikit-learn API:** estimators use `fit`, `predict` and `transform`, store
  fitted attributes with a trailing underscore, and pass scikit-learn's estimator checks.
- **Independent validation:** each method is tested against its definition,
  theoretical properties and published results. R packages such as robustbase
  and rrcov are an external reference, and differences in conventions are documented.
- **Explicit conventions:** choices such as finite-sample corrections or the
  subset size `h` are documented and, where useful, selectable.
- **Few dependencies:** NumPy, SciPy and scikit-learn.
- **Contribute upstream:** where a correct implementation already exists in
  another project, improve that project instead of duplicating it.

## Available

- **Location:** Hodges–Lehmann, M, tau, trimmed and winsorized means.
- **Scale, skewness and tail weight:** MAD, Qn, Sn, Pn, tau scale, M-scale,
  medcouple and its left and right versions.
- **Covariance:** MCD (FAST-MCD and DetMCD), MRCD, MVE, OGK, S and MM estimators.

## Next: robust regression

- LTS, S and MM regression.
- Diagnostics: robust residuals, leverage and outlier maps.

## Later stages

1. **Robust multivariate methods:** robust PCA (ROBPCA and projection pursuit),
   robust LDA and QDA, robust Hotelling's T² and MANOVA, and the
   Stahel–Donoho estimator.
2. **Multivariate density estimation:** kernel density estimation with a full
   bandwidth matrix (plug-in and smoothed cross-validation), density
   derivatives, highest density regions and two-sample tests.
3. **Robust generalized linear models:** Mallows quasi-likelihood, Bianco–Yohai,
   weighted Bianco–Yohai and MT estimators, and robust nonlinear regression.
4. **Depth and outlyingness:** projection depth (random directions, Peña–Prieto,
   skewness-adjusted), regression depth, the bagplot, and adjusted and
   directional outlyingness.
5. **Rank-based methods:** rank regression and exact conditional tests with ties.
6. **Shape-constrained estimation:** univariate log-concave density estimation
   and regression by antitonic score matching.

## Across all stages

- Diagnostic plots, such as the adjusted boxplot, distance–distance plot,
  tolerance ellipses and outlier maps.
- Classic datasets from the robust statistics literature, such as hbk,
  stackloss, starsCYG and wood.
- A table that maps R functions to their robustmodels equivalents, and a
  gallery of examples.

## Not planned for now

Nonlinear and censored quantile regression, quantile and robust mixed models,
sparse LTS, multivariate log-concave density estimation, and integrations with
distributed computing frameworks.
