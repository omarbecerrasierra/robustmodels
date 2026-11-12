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

## Differences to keep in mind

- **Data layout:** as in R, observations are rows and variables are columns.
- **Distances:** `dist_` and `mahalanobis` are squared distances, as in R's `mahalanobis`.
- **Random starts:** use `random_state=` instead of `set.seed()`. Searches with
  random starts (FAST-MCD, MVE, S) can select different subsets than R, so
  estimates may differ slightly. `MCD(algorithm="deterministic")` needs no seed.
- **Defaults:** some defaults differ from R, such as the consistency factor in
  OGK and the treatment of ties in the medcouple. The
  [method references](references.md) list each difference.
