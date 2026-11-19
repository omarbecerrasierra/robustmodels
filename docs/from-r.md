# Coming from R: robustbase and rrcov equivalents

This page maps functions from R's robustbase, rrcov and base packages to
robustmodels. These are migration correspondences, not promises of identical
results or a definition of correctness. The [method references](references.md)
and [validation notes](validation.md) describe mathematical definitions,
finite-sample corrections, tie handling and computational approximations.

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
`c = 1.345`; pass `c=1.5` to select that same tuning convention. `location.winsorized_mean` replaces
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
| `RobStatTM:::initPP(X)` (KSD) | `PenaPrieto().fit(X)` (covariance scaled for consistency) |
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
| `anova(fit, reduced_formula, test = "Deviance")` | `regression.mm_test(mm, X, y, columns)` |
| `anova(fit, reduced_formula, test = "Wald")` | `regression.mm_test(mm, X, y, columns, test="wald")` |

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
| `ddalpha::depth.halfspace(Z, X, exact = TRUE)` | `depth.halfspace_depth(X, Z, exact=True)` |
| `mrfDepth::rdepth(cbind(X, y), fits)` | `depth.regression_depth(X, y, coef, intercept)` for each fit |
| `mrfDepth::hdepthmedian(X)` | `depth.halfspace_median(X)` (centroid convention) |
| `mrfDepth::compBagplot(X)`, `aplpack::bagplot(X)` | `depth.bagplot_stats(X)`, `plots.bagplot(X)` |

See the [depth guide](depth.md) for approximations and center conventions.

## Rank-based methods

| R | robustmodels |
| --- | --- |
| `Rfit::rfit(y ~ X)` | `rank.RankRegressor().fit(X, y)` |
| `summary(rfit_fit)` | `summary()`, `std_errors_`, `t_values_`, `p_values_` |
| `Rfit::drop.test(full, reduced)` | `rank.drop_test(full, reduced)` |
| `Rfit::oneway.rfit(y, g)` | `rank.rank_anova(y, g)` (the overall test, without pairwise comparisons) |
| `exactRankTests::wilcox.exact(x, y)`, `coin::wilcox_test(..., distribution = "exact")` | `rank.rank_sum_test(x, y)` |
| `exactRankTests::wilcox.exact(x)`, `wilcox.exact(x, y, paired = TRUE)` | `rank.signed_rank_test(x)`, `rank.signed_rank_test(x, y)` |
| `coin::wilcoxsign_test(..., distribution = "exact")` | `rank.signed_rank_test(x, y, zero_method="pratt")` |

| `rfit` result | `RankRegressor` attribute |
| --- | --- |
| `coefficients` | `intercept_`, `coef_` |
| `tauhat`, `taushat` | `tau_`, `tau_s_` |
| `D1`, `vcov(fit)` | `dispersion_`, `cov_params_` |

The fit is the exact minimum of the dispersion, which Rfit approximates, and
`tau` takes its quantile exactly; see the [rank guide](rank.md). The test
statistics are `W` (Mann-Whitney form) and `V`, as `wilcox.exact`.

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

## Robust clustering

| R | robustmodels |
| --- | --- |
| `tclust::tclust(X, k, alpha)` | `cluster.TClust(k, alpha=alpha).fit(X)` |
| `tclust::tkmeans(X, k, alpha)` | `cluster.TClust(k, alpha=alpha, restr_factor=1, equal_weights=True).fit(X)` |
| `plot(tclust_fit)` | `plots.cluster_plot(model, X)` |
| `plot(tclust::ctlcurves(X))` | `plots.ctl_curves(X)` |

| `tclust` result | `TClust` attribute |
| --- | --- |
| `cluster` (0 for trimmed) | `labels_` (-1 for trimmed, zero-based clusters) |
| `centers`, `cov`, `weights` | `means_` (clusters in rows), `covariances_`, `weights_` |
| `obj` | `objective_` |

The `restr = "deter"` and `"GPCM"` restrictions, `opt = "MIXT"` and
`ctlcurves` are not implemented.

## Density estimation

| R | robustmodels |
| --- | --- |
| `ks::Hpi(X, pilot = "dscalar", deriv.order = r)` | `density.plugin_bandwidth(X, deriv_order=r)` |
| `ks::Hscv(X, pilot = "dscalar", deriv.order = r)` | `density.scv_bandwidth(X, deriv_order=r)` |
| `ks::kde(X)`, `predict(fhat, x = Y)` | `density.KernelDensity().fit(X)`, `np.exp(model.score_samples(Y))` |
| `ks::kdde(X, deriv.order = 1)` | `density.KernelDensity(deriv_order=1).fit(X).gradient(Y)` |
| `ks::kdde(X, deriv.order = 2)` | `density.KernelDensity(deriv_order=2).fit(X).hessian(Y)` |
| `ks::contourLevels(fhat, cont = 100 * p)` | `model.hdr_level(p)` |
| `ks::kde.test(X1, X2)` | `density.kde_test(X1, X2)` |
| `plot(fhat, display = "filled.contour")` | `plots.hdr_plot(model)` |

ks bins large samples and uses the SAMSE pilot for the density of two
variables by default; see the
[density guide](density.md#numerical-checks-and-sources) for these and the
other differences.

## Shape-constrained estimation

| R | robustmodels |
| --- | --- |
| `logcondens::logConDens(x, smoothed = FALSE)` | `shape.LogConcaveDensity().fit(x)` |
| `evaluateLogConDens(t, fit)` | `model.score_samples(t)`, `model.cdf(t)` |
| `plot(fit)` | `plots.log_concave_plot(model)` |
| `asm::asm(y ~ X)` | `shape.ScoreMatchingRegressor().fit(X, y)` |
| `asm::asm(y ~ X, symmetric = FALSE, intercept.selection = "median")` | `shape.ScoreMatchingRegressor(symmetric=False).fit(X, y)` |
| `summary(asm_fit)`, `plot(asm_fit, which = 3)` | `model.summary()`, `plots.score_plot(model, X, y)` |

logcondens's smoothed estimate is not implemented. asm's default intercept
for `symmetric = FALSE` is the mean of the residuals; robustmodels uses the
median.

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
