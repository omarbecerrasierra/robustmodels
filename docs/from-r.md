# Coming from R: robustbase and rrcov equivalents

This page maps functions from R's robustbase, rrcov and base packages to
robustmodels. Results agree up to the conventions documented in the
[method references](references.md), such as finite-sample corrections, tie
handling and random starts.

## Location and scale

| R | robustmodels |
| --- | --- |
| `mean(x, trim = 0.1)` | `location.trimmed_mean(x, 0.1)` |
| `DescTools::HodgesLehmann(x)` | `location.hodges_lehmann(x)` |
| `robustbase::huberM(x)` | `location.m_location(x, c=1.5)` |
| `robustbase::scaleTau2(x, mu.too = TRUE)` | `location.tau_location(x)`, `scale.tau_scale(x)` |
| `mad(x)` | `scale.mad(x)` |
| `robustbase::Qn(x)`, `robustbase::Sn(x)` | `scale.qn(x)`, `scale.sn(x)` |
| `robustbase::mc(x)` | `scale.medcouple(x)` |

`huberM` uses `k = 1.5` by default, while `m_location` uses the more common
`c = 1.345`; pass `c=1.5` to match R. `location.winsorized_mean` replaces
`floor(n * proportion)` observations in each tail, like `mean(x, trim =)`;
`psych::winsor.mean` and `DescTools::Winsorize` clip at sample quantiles
instead, so they can give different results. robustmodels also provides `scale.pn`
(Tarr's Pn), `scale.m_scale`, and the tail-weight measures `scale.lmc` and
`scale.rmc`, which have no exported equivalent in robustbase.

## Covariance

| R | robustmodels |
| --- | --- |
| `robustbase::covMcd(X)`, `rrcov::CovMcd(X)` | `MCD().fit(X)` |
| `robustbase::covMcd(X, nsamp = "deterministic")` | `MCD(algorithm="deterministic").fit(X)` |
| `rrcov::CovMrcd(X)` | `MRCD().fit(X)` |
| `rrcov::CovMve(X)`, `MASS::cov.mve(X)` | `MVE().fit(X)` |
| `robustbase::covOGK(X, sigmamu = scaleTau2)`, `rrcov::CovOgk(X)` | `OGK().fit(X)` |
| `rrcov::CovSde(X)` | `StahelDonoho().fit(X)` (another weight function) |
| `rrcov::CovSest(X)` | `SEstimator().fit(X)` |
| `rrcov::CovMMest(X)` | `MMEstimator().fit(X)` |
| `robustbase::h.alpha.n(alpha, n, p)` | `h_alpha_n(alpha, n, p)` |
| `mahalanobis(X, center, cov)` | `model.mahalanobis(X)` |

The estimators are classes in `robustmodels.covariance`. Fitted results are
attributes with a trailing underscore. For `covMcd`:

| `covMcd` result | robustmodels attribute |
| --- | --- |
| `center`, `cov` | `location_`, `covariance_` |
| `raw.center`, `raw.cov` | `raw_location_`, `raw_covariance_` |
| `mah` | `dist_` |
| `quan` | `h_` |
| `best` | `raw_support_` (a boolean mask instead of indices) |
| `mcd.wt` | `support_` |

## Regression

| R | robustmodels |
| --- | --- |
| `robustbase::lmrob(y ~ X)` | `regression.MMRegressor().fit(X, y)` |
| `robustbase::lmrob(y ~ X, method = "S")` | `regression.SRegressor().fit(X, y)` |
| `robustbase::ltsReg(X, y)` | `regression.LTSRegressor().fit(X, y)` |
| `robustbase::ltsReg(X, y, intercept = FALSE)` | `regression.LTSRegressor(fit_intercept=False).fit(X, y)` |

| `lmrob` result | `MMRegressor` attribute or method |
| --- | --- |
| `coefficients`, `scale` | `intercept_`, `coef_`, `scale_` |
| `init.S$coefficients` | `s_intercept_`, `s_coef_` |
| `vcov(fit)` | `cov_params_` |
| `summary(fit)` | `summary()`, `std_errors_`, `t_values_`, `p_values_` |

| `ltsReg` result | robustmodels attribute |
| --- | --- |
| `coefficients` | `intercept_`, `coef_` |
| `raw.coefficients` | `raw_intercept_`, `raw_coef_` |
| `scale`, `raw.scale` | `scale_`, `raw_scale_` |
| `quan` | `h_` |
| `best` | `raw_support_` (a boolean mask instead of indices) |
| `raw.weights` | `support_` |

## Generalized linear and nonlinear models

| R | robustmodels |
| --- | --- |
| `glmrob(..., family=binomial, method="Mqle")` | `glm.MallowsLogisticRegression().fit(X, y)` |
| `glmrob(..., family=poisson, method="Mqle")` | `glm.MallowsPoissonRegressor().fit(X, y)` |
| `glmrob(..., method="BY")` | `glm.BYLogisticRegression(weighted=False).fit(X, y)` |
| `glmrob(..., method="MT")` | `glm.MTPoissonRegressor(random_state=0).fit(X, y)` |
| `robustbase::nlrob(...)` | `regression.NonlinearMRegressor(model, start).fit(X, y)` |

Weighted BY trims throughout fitting here; R's `WBY` trims only for its start.
See the [GLM guide](glm.md) for leverage weights, covariance and MT interpolation
differences, and the [regression guide](regression.md) for nonlinear covariance.

## Depth

| R | robustmodels |
| --- | --- |
| `mrfDepth::hdepth(X, Z)` | `depth.halfspace_depth(X, Z)` |
| `mrfDepth::rdepth(cbind(X, y), fits)` | `depth.regression_depth(X, y, coef, intercept)` for each fit |
| `mrfDepth::hdepthmedian(X)` | `depth.halfspace_median(X)` (centroid convention) |
| `mrfDepth::compBagplot(X)`, `aplpack::bagplot(X)` | `depth.bagplot_stats(X)`, `plots.bagplot(X)` |

See the [depth guide](depth.md) for approximations and center conventions.

## Multivariate methods

| R | robustmodels |
| --- | --- |
| `rrcov::PcaHubert(X, k = 2)` | `multivariate.ROBPCA(2).fit(X)` |
| `pcaPP::PCAgrid(X, k = 2)`, `rrcov::PcaGrid(X, k = 2)` | `multivariate.ProjectionPursuitPCA(2).fit(X)` |
| `rrcov::Linda(X, y)` | `multivariate.RobustLDA().fit(X, y)` |
| `rrcov::QdaCov(X, y)` | `multivariate.RobustQDA().fit(X, y)` |
| `rrcov::T2.test(X, mu = mu, method = "mcd")` | `multivariate.hotelling_t2(X, mu)` |
| `rrcov::Wilks.test(X, grouping, method = "mcd")` | `multivariate.manova(X, groups)` |

| `PcaHubert` result | `ROBPCA` attribute or method |
| --- | --- |
| `center`, `loadings`, `eigenvalues` | `location_`, `components_` (axes in rows), `explained_variance_` |
| `scores` | `transform(X)` |
| `sd`, `od` | `score_distances_`, `orthogonal_distances_` |
| `cutoff.sd`, `cutoff.od` | `score_cutoff_`, `orthogonal_cutoff_` (univariate MCD instead of median and MAD) |
| `flag` | `support_` |

`ROBPCA` fixes the sign of each component (its largest loading is positive);
`PcaHubert` does not. `PCAgrid` and `ProjectionPursuitPCA` are local searches
and can stop at different directions. `Wilks.test` simulates its null
distribution with FAST-MCD and `manova` with the deterministic MCD, so the
statistic and p-value can differ; see the
[multivariate guide](multivariate.md#numerical-limits-and-sources).

## Diagnostic plots

| R | robustmodels |
| --- | --- |
| `plot(lmrob_fit, which = 1)` | `plots.outlier_map(model, X, y)` |
| `covPlot(X, which = "dd", m.cov = fit)` | `plots.distance_distance(model, X)` |
| `covPlot(X, which = "tolEllipsePlot", m.cov = fit, classic = TRUE)` | `plots.tolerance_ellipse(model, X)` |
| `robustbase::adjbox(X)` | `plots.adjusted_boxplot(X)` |
| `plot(lmrob_fit, which = 2)` | `plots.residual_qq(model, X, y)` |
| `plot(lmrob_fit, which = 4)` | `plots.residual_plot(model, X, y)` |
| `covPlot(X, which = "qqchi2", m.cov = fit)` | `plots.distance_qq(model, X)` |
| `plot(PcaHubert_fit)` | `plots.pca_outlier_map(model)` |
| `screeplot(PcaHubert_fit)`, `biplot(PcaHubert_fit)` | `plots.scree_plot(model)`, `plots.biplot(model, X)` |

These are corresponding diagnostics, not identical rendering routines.
Labels are zero-based in Python; see the [plot conventions](plots.md#sources-and-r-conventions)
for chi-square degrees of freedom, covariance scaling and medcouple ties.
`plots.outlyingness_plot(detector)` displays training scores and the fitted
cutoff from any projection detector; pass `X=` to score new observations.

## Outlier detection

| R | robustmodels |
| --- | --- |
| `mrfDepth::outlyingness(X)` | `outliers.StahelDonohoOutlyingness().fit(X)` |
| `mrfDepth::dirOutl(X)` | `outliers.DirectionalOutlyingness().fit(X)` |
| `mrfDepth::adjOutl(X)` | `outliers.AdjustedOutlyingness().fit(X)` |
| `mrfDepth::projdepth(X)`, `mrfDepth::sprojdepth(X)` | `score_samples(X)` of the detectors above |

| mrfDepth result | robustmodels attribute or method |
| --- | --- |
| `outlyingnessX` | `outlyingness_` |
| `outlyingnessZ` (new points `z`) | `outlyingness(Z)` |
| `cutoff` | `cutoff_` |
| `flagX` (TRUE for inliers) | `predict(X) == 1` |

`robustbase::adjOutlyingness` computes the same adjusted outlyingness with
type-7 quartiles (pass `IQRtype = 5` to match) and flags observations above the
adjusted boxplot fence of the outlyingness values instead of mrfDepth's cutoff.
mrfDepth's adjusted outlyingness can differ on data with tied projections; see
the [outlier detection guide](outliers.md).

## Differences to keep in mind

- **Data layout:** as in R, observations are rows and variables are columns.
- **Distances:** `dist_` and `mahalanobis` are squared distances, as in R's `mahalanobis`.
- **Random starts:** use `random_state=` instead of `set.seed()`. Searches with
  random starts (FAST-MCD, MVE, S) can select different subsets than R, so
  estimates may differ slightly. `MCD(algorithm="deterministic")` needs no seed.
- **Defaults:** some defaults differ from R, such as the consistency factor in
  OGK and the treatment of ties in the medcouple. The
  [method references](references.md) list each difference.
