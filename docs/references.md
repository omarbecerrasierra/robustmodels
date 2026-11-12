# Method references

These references describe the methods implemented in the **current**
modules. They are separate from citing the robustmodels software itself via
`CITATION.cff`. A citation identifies a definition or algorithm; it does not
certify this implementation. The method pages explain numerical conventions and limitations.

## Method-to-source map

| Public method | Scientific source | Implemented convention / R comparison |
| --- | --- | --- |
| `location.hodges_lehmann` | [HL63](#hl63) | Median of Walsh averages, including diagonal pairs; direct definition in R |
| `location.trimmed_mean`, `location.winsorized_mean` | [TM63](#tm63) | `floor(n * proportion)` observations per tail; R mean / explicit clipping |
| `location.m_location` | [H64](#h64), [BT74](#bt74) | Fixed scale, median start; Huber compared with `robustbase::huberM(k=1.5)`; bisquare checked by its score equation |
| `location.tau_location`, `scale.tau_scale` | [MZ02](#mz02), [YZ88](#yz88) | One-step tau form used by OGK, not a general tau regression estimator; `scaleTau2(iter=1)` |
| `scale.mad` | [H74](#h74), [CR92](#cr92) | Median center by default; exact normal quantile; finite correction optional; R `mad` uses rounded 1.4826 |
| `scale.qn`, `scale.sn` | [RC93](#rc93), [CR92](#cr92) | Qn order statistic and Sn low/high medians; finite corrections on by default; compare raw and corrected values separately |
| `scale.pn` | [TMW12](#tmw12), [T14](#t14) | Inverse empirical CDF (R quantile type 1) of off-diagonal pairwise means; author's finite factors |
| `scale.m_scale` | [MMYS19](#mmys19), [D87](#d87) | Normalized bisquare loss, fixed center, calibrated normal consistency; independently bracketed equation tests |
| `scale.medcouple` | [BHS04](#bhs04) | Tie kernel (2.3); average of middle kernels for even counts; can differ from R on tied samples |
| `scale.lmc`, `scale.rmc` | [BHS06](#bhs06) | Signed medcouple on the corresponding half sample |
| `covariance.MCD` | [R84](#r84), [RVD99](#rvd99), [HRV12](#hrv12) | Random/deterministic C-step searches; exact sliding-window search in one dimension; `covMcd` |
| `covariance.h_alpha_n` | [HRV12](#hrv12), [robustbase](#r-software) | Same floor-based interpolation as `robustbase::h.alpha.n` |
| `covariance.mcd_consistency` | [CH99](#ch99) | Gaussian truncated second-moment correction |
| `covariance.mcd_small_sample` | [PVAW02](#pvaw02) | Empirical fitted finite-sample factors; not exact unbiasedness |
| `covariance.MRCD` | [BRVV20](#brvv20) | Identity target in robustly standardized coordinates; common rho from algorithm step 3.4; `CovMrcd` |
| `covariance.MVE` | [R85](#r85), [VAR09](#var09) | Random elemental ellipsoids and local refits; approximate search, not a certified global MVE; `CovMve` is diagnostic only |
| `covariance.OGK` | [GK72](#gk72), [MZ02](#mz02) | Adds a Gaussian consistency factor to reweighted scatter, unlike `covOGK`; raw estimates also compared |
| `covariance.SEstimator` | [D87](#d87), [SBVAW06](#sbvaw06), [HRVV15](#hrvv15) | FAST-S style subsampling and refinement; `CovSest(method="sfast")` |
| `regression.LTSRegressor` | [R84](#r84), [RVD06](#rvd06), [PVAW02](#pvaw02) | FAST-LTS without intercept adjustment; 0/1 reweighting at the 0.9875 normal quantile; reweighted scale divides by `sum(w) - 1` and matches `ltsReg` |
| `outliers.StahelDonohoOutlyingness` | [S81](#s81), [D82](#d82), [Z03](#z03) | Directions normal to hyperplanes through `p` points (all when few, else random), consistent MAD, log cutoff as mrfDepth; `outlyingness`, `projdepth` |
| `outliers.AdjustedOutlyingness` | [HVdV08](#hvdv08), [HV08](#hv08) | Adjusted boxplot whiskers with type-5 quartiles and the medcouple by definition; ties from rounding merged; log cutoff as mrfDepth; `adjOutl`, `sprojdepth` |
| `regression.SRegressor` | [RY84](#ry84), [SBY06](#sby06), [MMYS19](#mmys19) | FAST-S with five candidates refined; M-scale divides by `n - p`; exact bisquare constant (`lmrob` rounds to 1.54764); `lmrob(method = "S")` |
| `regression.MMRegressor` | [Y87](#y87), [SBY06](#sby06), [CDH03](#cdh03) | S scale fixed; bisquare at 95% coefficient efficiency (exact constant; `lmrob` rounds to 4.685061); covariance as `lmrob`'s `.vcov.avar1`, with the S breakdown in the scale term (`lmrob` fixes 1/2); `lmrob` default |
| `covariance.MMEstimator` | [TT00](#tt00), [SBVAW06](#sbvaw06), [HRVV15](#hrvv15) | S scale held fixed, shape efficiency tuning, ordered normalized losses; `CovMMest(eff.shape=TRUE)` |

## Bibliography

(hl63)=
### HL63
Hodges, J. L., Jr. and Lehmann, E. L. (1963). *Estimates of Location Based on
Rank Tests*. Annals of Mathematical Statistics, **34**(2), 598–611.
[DOI](https://doi.org/10.1214/aoms/1177704172).

(tm63)=
### TM63
Tukey, J. W. and McLaughlin, D. H. (1963). *Less Vulnerable Confidence and
Significance Procedures for Location Based on a Single Sample:
Trimming/Winsorization 1*. Sankhya, Series A, **25**(3), 331–352.
[Journal issue](https://www.jstor.org/stable/i25049268).

(h64)=
### H64
Huber, P. J. (1964). *Robust Estimation of a Location Parameter*.
Annals of Mathematical Statistics, **35**(1), 73–101.
[DOI](https://doi.org/10.1214/aoms/1177703732).

(bt74)=
### BT74
Beaton, A. E. and Tukey, J. W. (1974). *The Fitting of Power Series, Meaning
Polynomials, Illustrated on Band-Spectroscopic Data*. Technometrics,
**16**(2), 147–185. [DOI](https://doi.org/10.1080/00401706.1974.10489171).

(h74)=
### H74
Hampel, F. R. (1974). *The Influence Curve and Its Role in Robust Estimation*.
Journal of the American Statistical Association, **69**(346), 383–393.
[DOI](https://doi.org/10.1080/01621459.1974.10482962).

(cr92)=
### CR92
Croux, C. and Rousseeuw, P. J. (1992). *Time-Efficient Algorithms for Two
Highly Robust Estimators of Scale*. In Computational Statistics 1992, Vol. 1,
411–428. Physica-Verlag. The factors are numerical approximations, not
finite-sample identities; modern Qn factors follow the
[robustbase implementation](https://github.com/cran/robustbase/blob/master/R/qnsn.R).

(rc93)=
### RC93
Rousseeuw, P. J. and Croux, C. (1993). *Alternatives to the Median Absolute
Deviation*. Journal of the American Statistical Association, **88**(424),
1273–1283. [Author-hosted article](https://wis.kuleuven.be/stat/robust/papers/publications-1993/rousseeuwcroux-alternativestomedianad-jasa-1993.pdf).

(tmw12)=
### TMW12
Tarr, G., Müller, S. and Weber, N. C. (2012). *A Robust Scale Estimator Based
on Pairwise Means*. Journal of Nonparametric Statistics, **24**(1), 187–199.
[Publisher and DOI](https://www.tandfonline.com/doi/abs/10.1080/10485252.2011.621424).
[Author's R reference and finite factors](https://www.maths.usyd.edu.au/u/gartht/R/Pn.R).

(t14)=
### T14
Tarr, G. (2014). *Quantile Based Estimation of Scale and Dependence*.
PhD thesis, University of Sydney. [Permanent record](https://hdl.handle.net/2123/10590).
The finite correction factors are also available in the author's R reference
linked under TMW12.

(yz88)=
### YZ88
Yohai, V. J. and Zamar, R. H. (1988). *High Breakdown-Point Estimates of
Regression by Means of the Minimization of an Efficient Scale*.
Journal of the American Statistical Association, **83**(402), 406–413.
[DOI](https://doi.org/10.1080/01621459.1988.10478611).

(mz02)=
### MZ02
Maronna, R. A. and Zamar, R. H. (2002). *Robust Estimates of Location and
Dispersion for High-Dimensional Datasets*. Technometrics, **44**(4), 307–317.
[DOI](https://doi.org/10.1198/004017002188618509).
The OGK algorithm is also reproduced in Appendix B of [BRVV20](#brvv20).
The tau formula is documented in `robustbase::scaleTau2`.

(bhs04)=
### BHS04
Brys, G., Hubert, M. and Struyf, A. (2004). *A Robust Measure of Skewness*.
Journal of Computational and Graphical Statistics, **13**(4), 996–1017.
[Author-hosted preprint](https://wis.kuleuven.be/stat/robust/papers/2004/medcouple.pdf).
The linked preprint is dated September 1, 2003; definitions (2.1)–(2.3) specify the
sample median and the tie kernel.

(bhs06)=
### BHS06
Brys, G., Hubert, M. and Struyf, A. (2006). *Robust Measures of Tail Weight*.
Computational Statistics & Data Analysis, **50**(3), 733–759.
[Author-hosted manuscript](https://wis.kuleuven.be/stat/robust/papers/2006/tailweight.pdf).

(r84)=
### R84
Rousseeuw, P. J. (1984). *Least Median of Squares Regression*.
Journal of the American Statistical Association, **79**(388), 871–880.
[DOI](https://doi.org/10.1080/01621459.1984.10477105).
Introduces LTS regression and the MCD; this package does not implement LMS regression.

(r85)=
### R85
Rousseeuw, P. J. (1985). *Multivariate Estimation with High Breakdown Point*.
In Mathematical Statistics and Applications, Vol. B, 283–297. Reidel.
See the detailed treatment and bibliography in [VAR09](#var09).

(rvd99)=
### RVD99
Rousseeuw, P. J. and Van Driessen, K. (1999). *A Fast Algorithm for the
Minimum Covariance Determinant Estimator*. Technometrics, **41**(3), 212–223.
[Author-hosted article](https://wis.kuleuven.be/stat/robust/papers/1999/rousseeuwvandriessen-fastalgorithmformcd-technomet.pdf).

(s81)=
### S81
Stahel, W. A. (1981). *Robuste Schätzungen: infinitesimale Optimalität und
Schätzungen von Kovarianzmatrizen*. PhD thesis, ETH Zürich.

(d82)=
### D82
Donoho, D. L. (1982). *Breakdown Properties of Multivariate Location
Estimators*. Qualifying paper, Department of Statistics, Harvard University.

(z03)=
### Z03
Zuo, Y. (2003). *Projection-Based Depth Functions and Associated Medians*.
The Annals of Statistics, **31**(5), 1460–1490.
[DOI](https://doi.org/10.1214/aos/1065705115).

(hvdv08)=
### HVdV08
Hubert, M. and Van der Veeken, S. (2008). *Outlier Detection for Skewed Data*.
Journal of Chemometrics, **22**(3–4), 235–246.
[DOI](https://doi.org/10.1002/cem.1123).

(hv08)=
### HV08
Hubert, M. and Vandervieren, E. (2008). *An Adjusted Boxplot for Skewed
Distributions*. Computational Statistics & Data Analysis, **52**(12),
5186–5201. [DOI](https://doi.org/10.1016/j.csda.2007.11.008).

(ry84)=
### RY84
Rousseeuw, P. J. and Yohai, V. J. (1984). *Robust Regression by Means of
S-Estimators*. In Robust and Nonlinear Time Series Analysis, Lecture Notes in
Statistics 26, 256–272. Springer.
[DOI](https://doi.org/10.1007/978-1-4615-7821-5_15).

(sby06)=
### SBY06
Salibian-Barrera, M. and Yohai, V. J. (2006). *A Fast Algorithm for
S-Regression Estimates*. Journal of Computational and Graphical Statistics,
**15**(2), 414–427. [DOI](https://doi.org/10.1198/106186006X113629).

(y87)=
### Y87
Yohai, V. J. (1987). *High Breakdown-Point and High Efficiency Robust Estimates
for Regression*. The Annals of Statistics, **15**(2), 642–656.
[DOI](https://doi.org/10.1214/aos/1176350366).

(cdh03)=
### CDH03
Croux, C., Dhaene, G. and Hoorelbeke, D. (2003). *Robust Standard Errors for
Robust Estimators*. Discussion Paper 03.16, Center for Economic Studies,
KU Leuven. [RePEc](https://ideas.repec.org/p/ete/ceswps/ces0316.html).

(rvd06)=
### RVD06
Rousseeuw, P. J. and Van Driessen, K. (2006). *Computing LTS Regression for
Large Data Sets*. Data Mining and Knowledge Discovery, **12**(1), 29–45.
[DOI](https://doi.org/10.1007/s10618-005-0024-4).

(hrv12)=
### HRV12
Hubert, M., Rousseeuw, P. J. and Verdonck, T. (2012). *A Deterministic
Algorithm for Robust Location and Scatter*. Journal of Computational and
Graphical Statistics, **21**(3), 618–637.
[Author-hosted manuscript](https://wis.kuleuven.be/stat/robust/papers/2012/DetMCD.pdf).

(ch99)=
### CH99
Croux, C. and Haesbroeck, G. (1999). *Influence Function and Efficiency of
the Minimum Covariance Determinant Scatter Matrix Estimator*.
Journal of Multivariate Analysis, **71**(2), 161–190.
[DOI](https://doi.org/10.1006/jmva.1999.1839).

(pvaw02)=
### PVAW02
Pison, G., Van Aelst, S. and Willems, G. (2002). *Small Sample Corrections
for LTS and MCD*. Metrika, **55**, 111–123.
[Author-hosted manuscript](https://wis.kuleuven.be/statdatascience/robust/papers/2002/PisonVanAelstWillems-SSC-Metrika2002.pdf).

(brvv20)=
### BRVV20
Boudt, K., Rousseeuw, P. J., Vanduffel, S. and Verdonck, T. (2020).
*The Minimum Regularized Covariance Determinant Estimator*.
Statistics and Computing, **30**, 113–128.
[Institutional full text](https://research.vu.nl/ws/portalfiles/portal/123663555/Boudt2020_The_minimum_regularized_covariance_determinant_estimator.pdf).
The implemented common-rho rule is algorithm step 3.4; the target is restricted
to identity in standardized coordinates.

(var09)=
### VAR09
Van Aelst, S. and Rousseeuw, P. J. (2009). *Minimum Volume Ellipsoid*.
WIREs Computational Statistics, **1**, 71–82.
[Author-hosted article](https://wis.kuleuven.be/stat/robust/papers/2009/vanaelstrousseeuw-minimumvolumeellipsoid-wires-200.pdf).

(gk72)=
### GK72
Gnanadesikan, R. and Kettenring, J. R. (1972). *Robust Estimates, Residuals,
and Outlier Detection with Multiresponse Data*. Biometrics, **28**(1), 81–124.
[DOI](https://doi.org/10.2307/2528963).

(d87)=
### D87
Davies, P. L. (1987). *Asymptotic Behaviour of S-Estimates of Multivariate
Location Parameters and Dispersion Matrices*. Annals of Statistics,
**15**(3), 1269–1292. [Institutional full text](https://pure.tue.nl/ws/files/1669278/Metis214121.pdf).

(tt00)=
### TT00
Tatsuoka, K. S. and Tyler, D. E. (2000). *On the Uniqueness of S-Functionals
and M-Functionals under Nonelliptical Distributions*. Annals of Statistics,
**28**(4), 1219–1243. [Institutional record](https://www.researchwithrutgers.org/en/publications/on-the-uniqueness-of-s-functionals-and-m-functionals-under-nonell/).

(sbvaw06)=
### SBVAW06
Salibián-Barrera, M., Van Aelst, S. and Willems, G. (2006). *Principal
Components Analysis Based on Multivariate MM Estimators with Fast and Robust
Bootstrap*. Journal of the American Statistical Association, **101**(475),
1198–1211. [DOI](https://doi.org/10.1198/016214506000000096).
The multivariate S/MM fitting method is used here; PCA and bootstrap are not
implemented. The fitting steps and loss-ordering condition are also described
in [HRVV15](#hrvv15).

(hrvv15)=
### HRVV15
Hubert, M., Rousseeuw, P., Vanpaemel, D. and Verdonck, T. (2015).
*The DetS and DetMM Estimators for Multivariate Location and Scatter*.
Computational Statistics & Data Analysis, **81**, 64–75.
[Author-hosted article](https://wis.kuleuven.be/statdatascience/robust/papers/2015/hubertrousseeuwvanpaemelverdonck-detsanddetmm-csda.pdf).
Sections 2 and 5 explain the S/MM equations and ordered normalized losses.
DetS/DetMM are not implemented.

(mmys19)=
### MMYS19
Maronna, R. A., Martin, R. D., Yohai, V. J. and Salibián-Barrera, M. (2019).
*Robust Statistics: Theory and Methods (with R)*, 2nd edition. Wiley.
[DOI](https://doi.org/10.1002/9781119214656). 

(r-software)=
## R software

Package manuals and source identify the software conventions:
[robustbase](https://cran.r-project.org/package=robustbase),
[rrcov](https://cran.r-project.org/package=rrcov),
[mrfDepth](https://cran.r-project.org/package=mrfDepth).
The stored regression fixtures in `tests/data/r_reference.json` record the
versions that produced those values. Agreement with these fixtures is limited
to their data and settings, and does not imply universal agreement with R.
