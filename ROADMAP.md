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
- **Few dependencies:** NumPy, SciPy, scikit-learn and Matplotlib for diagnostics.
- **Contribute upstream:** where a correct implementation already exists in
  another project, improve that project instead of duplicating it.

## Available

- **Location:** Hodges–Lehmann, M, tau, trimmed and winsorized means.
- **Scale, skewness and tail weight:** MAD, Qn, Sn, Pn, tau scale, M-scale,
  medcouple and its left and right versions.
- **Covariance:** MCD (FAST-MCD and DetMCD), MRCD, MVE, OGK, Stahel–Donoho, S
  and MM estimators.
- **Multivariate methods:** robust PCA (ROBPCA and projection pursuit), robust
  LDA and QDA, robust one-sample Hotelling's T² and one-way MANOVA.
- **Regression:** least trimmed squares (FAST-LTS) with reweighting, and S and
  MM estimators (FAST-S), with standard errors and tests for MM; nonlinear
  Huber M regression.
- **Generalized linear models:** Mallows logistic and Poisson, Bianco–Yohai
  logistic (including leverage trimming), and MT Poisson regression.
- **Outlier detection and depth:** Stahel-Donoho, adjusted and directional
  outlyingness, projection depth, halfspace and regression depth, the Tukey
  median and the bagplot.
- **Diagnostics:** regression and PCA outlier maps, residual plots, quantile
  plots of residuals and robust distances, distance-distance plots, tolerance ellipses, scree
  plots, biplots, classification regions, adjusted boxplots, bagplots and outlyingness plots.

## Later stages

1. **Multivariate density estimation:** kernel density estimation with a full
   bandwidth matrix (plug-in and smoothed cross-validation), density
   derivatives, highest density regions and two-sample tests.
2. **Depth and outlyingness:** Peña–Prieto projection directions and further
   high-dimensional depth algorithms.
3. **Rank-based methods:** rank regression and exact conditional tests with ties.
4. **Shape-constrained estimation:** univariate log-concave density estimation
   and regression by antitonic score matching.

## Across all stages

- Extend diagnostic plots as new estimators become available.
- Classic datasets from the robust statistics literature, such as hbk,
  stackloss, starsCYG and wood.
- Keep the R-equivalents table and executable diagnostic gallery current
  as methods are added.

## Not planned for now

Nonlinear and censored quantile regression, quantile and robust mixed models,
sparse LTS, multivariate log-concave density estimation, and integrations with
distributed computing frameworks.
