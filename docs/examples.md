# Examples with classic data

These data sets appear throughout the robust statistics literature, and their
outliers are documented. Each example checks that the estimators find them.
The data ship with the package in `robustmodels.datasets`; each loader returns
the predictors `X`, the response or class labels `y` and the column names.
Row numbers are zero-based, so row 10 is the eleventh observation.

```pycon
>>> from robustmodels import datasets
>>> stars = datasets.load_stars()
>>> stars.X.shape, stars.feature_names, stars.target_name
((47, 1), ('log.Te',), 'log.light')

```

## Stars: leverage points reverse a slope

The Hertzsprung–Russell diagram of the star cluster CYG OB1 plots the
logarithm of the light intensity of 47 stars against the logarithm of their
surface temperature ([Rousseeuw and Leroy](references.md#rl87), chapter 2).
Most stars lie on the main sequence, where brighter stars are hotter. Four red
giants are cool and bright. They pull the least squares line until its slope
is negative. MM regression follows the main sequence and gives the giants
large residuals.

```pycon
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from robustmodels.regression import MMRegressor
>>> X, y = stars.X, stars.y
>>> round(float(LinearRegression().fit(X, y).coef_[0]), 2)
-0.41
>>> mm = MMRegressor(random_state=0).fit(X, y)
>>> round(float(mm.coef_[0]), 2)
2.25
>>> np.flatnonzero(np.abs(y - mm.predict(X)) / mm.scale_ > 2.5).tolist()
[10, 19, 29, 33]

```

The summary gives the MM estimates with their standard errors and the robust
residual scale. The positive slope is clearly significant.

```pycon
>>> print(mm.summary(["log.Te"]))
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)      -4.969        3.41    -1.457  0.152
log.Te            2.253      0.7691     2.930  0.00531
Residual scale: 0.4715

```

The scatter plot shows the three fits. Least squares passes between the main
sequence and the giants; LTS and MM follow the main sequence.

```{eval-rst}
.. plot::
   :filename-prefix: examples-stars-lines
   :include-source: true
   :alt: Star data with the least squares, LTS and MM regression lines.

   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn.linear_model import LinearRegression
   from robustmodels import datasets
   from robustmodels.regression import LTSRegressor, MMRegressor

   X, y, *_ = datasets.load_stars()
   grid = np.linspace(3.4, 4.7, 2)[:, None]
   fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
   ax.scatter(X[:, 0], y, s=18, color="0.45", label="Stars")
   ax.scatter(X[[10, 19, 29, 33], 0], y[[10, 19, 29, 33]], s=36, marker="D",
              color="#e8590c", label="Red giants (10, 19, 29, 33)")
   fits = [
       ("Least squares", LinearRegression().fit(X, y), "--"),
       ("LTS", LTSRegressor(random_state=0).fit(X, y), "-."),
       ("MM", MMRegressor(random_state=0).fit(X, y), "-"),
   ]
   for label, model, style in fits:
       ax.plot(grid[:, 0], model.predict(grid), style, lw=1.6, label=label)
   ax.set(xlabel="log.Te (surface temperature)", ylabel="log.light (light intensity)",
          ylim=(3.5, 6.6))
   ax.legend(fontsize=8, loc="lower left")
```

In the regression outlier map the giants are bad leverage points: extreme in
temperature and far from the fitted line.

```{eval-rst}
.. plot::
   :filename-prefix: examples-stars
   :include-source: true
   :alt: Regression outlier map of the star data with four red giants as bad leverage points.

   from robustmodels import datasets, plots
   from robustmodels.regression import MMRegressor

   X, y, *_ = datasets.load_stars()
   model = MMRegressor(random_state=0).fit(X, y)
   ax = plots.outlier_map(model, X, y, random_state=0)
```

The two variables can also be treated as a bivariate sample, without choosing
a response. The MCD tolerance ellipse follows the main sequence, while the
classical ellipse is inflated toward the giants. The bagplot, built from
halfspace depth, puts the four giants outside its fence.

```{eval-rst}
.. plot::
   :filename-prefix: examples-stars-bivariate
   :include-source: true
   :alt: Robust and classical tolerance ellipses beside a bagplot of the star data.

   import matplotlib.pyplot as plt
   import numpy as np
   from robustmodels import datasets, plots
   from robustmodels.covariance import MCD

   X, y, *_ = datasets.load_stars()
   Z = np.column_stack([X, y])
   names = ["log.Te", "log.light"]
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.tolerance_ellipse(MCD(random_state=0).fit(Z), Z, feature_names=names, ax=axes[0])
   plots.bagplot(Z, feature_names=names, ax=axes[1])
```

The ellipse also flags three stars at the edge of the main sequence, because
its 97.5% contour is a normal reference, while the bagplot fence is wider.

## Hawkins–Bradu–Kass: masking

[Hawkins, Bradu and Kass](references.md#hbk84) constructed 75 observations
with three predictors. Rows 0 to 13 are leverage points: rows 0 to 9 are bad
leverage points and rows 10 to 13 lie on the regression of the others. The
classical distances miss them, because the leverage points inflate the
covariance they are measured with. The MCD identifies all fourteen, and MM
regression separates the ten that are off the line.

```pycon
>>> from robustmodels.covariance import MCD
>>> X, y, *_ = datasets.load_hbk()
>>> mcd = MCD(random_state=0).fit(X)
>>> np.flatnonzero(~mcd.support_).tolist()
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
>>> mm = MMRegressor(random_state=0).fit(X, y)
>>> np.flatnonzero(np.abs(y - mm.predict(X)) / mm.scale_ > 2.5).tolist()
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

```

Hawkins, Bradu and Kass generated the regular rows without any relation
between the response and the predictors. The MM fit recovers this: no slope
is significant, and the residual scale is that of the regular rows.

```pycon
>>> print(mm.summary(["X1", "X2", "X3"]))
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)     -0.1896      0.1167    -1.624  0.109
X1              0.08527     0.07329     1.164  0.248
X2              0.04101     0.02956     1.387  0.17
X3             -0.05371     0.03195    -1.681  0.0971
Residual scale: 0.7892
>>> LinearRegression().fit(X, y).coef_.round(3).tolist()
[0.239, -0.335, 0.383]

```

Least squares instead reports three sizeable slopes, produced by the ten bad
leverage points.

```{eval-rst}
.. plot::
   :filename-prefix: examples-hbk
   :include-source: true
   :alt: Distance-distance plot and regression outlier map of the Hawkins-Bradu-Kass data.

   import matplotlib.pyplot as plt
   from robustmodels import datasets, plots
   from robustmodels.covariance import MCD
   from robustmodels.regression import MMRegressor

   X, y, *_ = datasets.load_hbk()
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.distance_distance(MCD(random_state=0).fit(X), X, ax=axes[0])
   plots.outlier_map(MMRegressor(random_state=0).fit(X, y), X, y, random_state=0, ax=axes[1])
```

On the left, the fourteen leverage points have small classical distances and
large robust ones. On the right, rows 0 to 9 are bad leverage points and rows
10 to 13 good leverage points.

The robust distances of the regular rows follow the chi-square reference, and
the leverage points sit far above it. Adjusted outlyingness, which needs no
covariance estimate and allows for skewness, flags the same fourteen rows.

```pycon
>>> from robustmodels.outliers import AdjustedOutlyingness
>>> ao = AdjustedOutlyingness(random_state=0).fit(X)
>>> np.flatnonzero(ao.outlyingness_ > ao.cutoff_).tolist()
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

```

```{eval-rst}
.. plot::
   :filename-prefix: examples-hbk-distances
   :include-source: true
   :alt: Chi-square quantile plot of robust distances and adjusted outlyingness of the Hawkins-Bradu-Kass predictors.

   import matplotlib.pyplot as plt
   from robustmodels import datasets, plots
   from robustmodels.covariance import MCD
   from robustmodels.outliers import AdjustedOutlyingness

   X, y, *_ = datasets.load_hbk()
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.distance_qq(MCD(random_state=0).fit(X), X, ax=axes[0])
   plots.outlyingness_plot(AdjustedOutlyingness(random_state=0).fit(X), ax=axes[1])
```

## Wood gravity: outliers in no single variable

[Rousseeuw](references.md#r84) replaced four of the twenty wood samples of
[Draper and Smith](references.md#ds66) by outliers. None of them is extreme
in any one of the five predictors, so marginal checks do not reveal them.
MM regression of the specific gravity on the five measurements does.

```pycon
>>> X, y, *_ = datasets.load_wood()
>>> mm = MMRegressor(random_state=0).fit(X, y)
>>> np.flatnonzero(np.abs(y - mm.predict(X)) / mm.scale_ > 2.5).tolist()
[3, 5, 7, 18]
>>> print(mm.summary(["x1", "x2", "x3", "x4", "x5"]))
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)      0.3784     0.04399     8.602  5.83e-07
x1               0.2165     0.03184     6.801  8.59e-06
x2             -0.08077      0.1422    -0.568  0.579
x3              -0.5639     0.02017   -27.962  1.1e-13
x4              -0.3982     0.05332    -7.467  3.03e-06
x5               0.6046     0.07546     8.013  1.34e-06
Residual scale: 0.01352

```

```{eval-rst}
.. plot::
   :filename-prefix: examples-wood
   :include-source: true
   :alt: Skewness-adjusted boxplots of the five wood predictors beside the normal quantile plot of MM residuals.

   import matplotlib.pyplot as plt
   from robustmodels import datasets, plots
   from robustmodels.regression import MMRegressor

   X, y, names, _ = datasets.load_wood()
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.adjusted_boxplot(X, labels=names, ax=axes[0])
   plots.residual_qq(MMRegressor(random_state=0).fit(X, y), X, y, ax=axes[1])
```

The boxplots show the predictors one at a time; the residuals of the robust
fit expose the four samples.

The outliers are also visible without the response. Robust principal
components of the five measurements leave the four samples far from the
principal subspace: they are orthogonal outliers, which break the correlation
of the regular samples without being extreme in any direction of it.

```pycon
>>> from robustmodels.multivariate import ROBPCA
>>> pca = ROBPCA(random_state=0).fit(X)
>>> pca.n_components_
3
>>> np.flatnonzero(pca.orthogonal_distances_ > pca.orthogonal_cutoff_).tolist()
[3, 5, 7, 18]

```

```{eval-rst}
.. plot::
   :filename-prefix: examples-wood-pca
   :include-source: true
   :alt: PCA outlier map of the wood predictors beside the MM residuals against fitted values.

   import matplotlib.pyplot as plt
   from robustmodels import datasets, plots
   from robustmodels.multivariate import ROBPCA
   from robustmodels.regression import MMRegressor

   X, y, *_ = datasets.load_wood()
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.pca_outlier_map(ROBPCA(random_state=0).fit(X), ax=axes[0])
   plots.residual_plot(MMRegressor(random_state=0).fit(X, y), X, y, ax=axes[1])
```

The scree plot shows how the robust variance is spread over the components,
and the biplot which measurements move together in the first plane.

```{eval-rst}
.. plot::
   :filename-prefix: examples-wood-biplot
   :include-source: true
   :alt: Scree plot of five robust components and biplot of the wood measurements.

   import matplotlib.pyplot as plt
   from robustmodels import datasets, plots
   from robustmodels.multivariate import ROBPCA

   X, y, names, _ = datasets.load_wood()
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.scree_plot(ROBPCA(5, random_state=0).fit(X), ax=axes[0])
   plots.biplot(ROBPCA(2, random_state=0).fit(X), X, feature_names=names, ax=axes[1])
```

## Stack loss: comparing fits

[Brownlee's](references.md#b65) stack loss data record 21 days of a plant
that oxidizes ammonia: air flow, cooling water temperature and acid
concentration, and the percentage of ammonia lost. Least squares weights all
days equally. LTS finds the four outlying days of
[Rousseeuw and Leroy](references.md#rl87), chapter 3, and its water
temperature coefficient is less than half that of least squares.

```pycon
>>> from robustmodels.regression import LTSRegressor
>>> X, y, *_ = datasets.load_stackloss()
>>> LinearRegression().fit(X, y).coef_.round(3).tolist()
[0.716, 1.295, -0.152]
>>> lts = LTSRegressor(random_state=0).fit(X, y)
>>> lts.coef_.round(3).tolist()
[0.798, 0.577, -0.067]
>>> np.flatnonzero(np.abs(y - lts.predict(X)) / lts.scale_ > 2.5).tolist()
[0, 2, 3, 20]

```

MM regression keeps the high breakdown of LTS and adds standard errors. Its
summary agrees with LTS on air flow and water temperature, and `mm_test`
compares the model with and without acid concentration by a robust deviance.

```pycon
>>> from robustmodels.regression import mm_test
>>> mm = MMRegressor(random_state=0).fit(X, y)
>>> print(mm.summary(["Air.Flow", "Water.Temp", "Acid.Conc."]))
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)      -41.52       5.298    -7.838  4.82e-07
Air.Flow         0.9388      0.1174     7.995  3.68e-07
Water.Temp       0.5796       0.263     2.204  0.0416
Acid.Conc.      -0.1129     0.06989    -1.616  0.125
Residual scale: 1.912
>>> round(mm_test(mm, X, y, [2]).pvalue, 2)
0.21

```

```{eval-rst}
.. plot::
   :filename-prefix: examples-stackloss
   :include-source: true
   :alt: Coefficients of MM and rank regression with 95% intervals against least squares for the stack loss data.

   from robustmodels import datasets, plots
   from robustmodels.rank import RankRegressor
   from robustmodels.regression import MMRegressor

   X, y, names, _ = datasets.load_stackloss()
   fits = [MMRegressor(random_state=0).fit(X, y), RankRegressor().fit(X, y)]
   ax = plots.coefficient_plot(fits, labels=["MM", "Rank"], X=X, y=y, feature_names=names)
```

The plot compares the fits that report standard errors. MM halves the water
temperature coefficient of least squares; rank regression, which resists
outliers in the response but not leverage points, lies between them. With 21
observations the intervals are wide: the plot shows where the estimators
disagree, not which one is right.

Rank regression gives a table of coefficients and a drop test of nested
models. Acid concentration does not improve the fit once air flow and water
temperature are in the model.

```pycon
>>> from robustmodels.rank import RankRegressor, drop_test
>>> names = list(datasets.load_stackloss().feature_names)
>>> full = RankRegressor().fit(X, y)
>>> print(full.summary(names))
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)      -40.12       11.29    -3.554  0.00244
Air.Flow         0.7917      0.1278     6.194  9.81e-06
Water.Temp       0.9111      0.3488     2.612  0.0182
Acid.Conc.      -0.1111      0.1481    -0.750  0.463
Drop in dispersion test: F = 28.71 on 3 and 17 degrees of freedom, p-value 7.01e-07
Robust R-squared: 0.8352
>>> round(drop_test(full, RankRegressor().fit(X[:, :2], y)).pvalue, 2)
0.39

```

The residuals of the LTS fit separate the four outlying days from the rest,
and the normal quantile plot shows that the remaining days are compatible
with normal errors.

```{eval-rst}
.. plot::
   :filename-prefix: examples-stackloss-residuals
   :include-source: true
   :alt: LTS residuals against fitted values and their normal quantile plot for the stack loss data.

   import matplotlib.pyplot as plt
   from robustmodels import datasets, plots
   from robustmodels.regression import LTSRegressor

   X, y, *_ = datasets.load_stackloss()
   model = LTSRegressor(random_state=0).fit(X, y)
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.residual_plot(model, X, y, ax=axes[0])
   plots.residual_qq(model, X, y, ax=axes[1])
```

## Hemophilia: robust discriminant analysis

[Habbema, Hermans and van den Broek](references.md#hhb74) measured AHF
activity and AHF antigen, on a log scale, in 30 women known not to carry
hemophilia A and 45 carriers. Robust linear discriminant analysis estimates
each group's center and a common covariance with the MCD.

```pycon
>>> from robustmodels.multivariate import RobustLDA, RobustQDA
>>> X, y, *_ = datasets.load_hemophilia()
>>> lda = RobustLDA(random_state=0).fit(X, y)
>>> qda = RobustQDA(random_state=0).fit(X, y)
>>> round(float(np.mean(lda.predict(X) != y)), 2), round(float(np.mean(qda.predict(X) != y)), 2)
(0.16, 0.16)

```

```{eval-rst}
.. plot::
   :filename-prefix: examples-hemophilia
   :include-source: true
   :alt: Robust linear discriminant regions for normal women and hemophilia A carriers.

   from robustmodels import datasets, plots
   from robustmodels.multivariate import RobustLDA

   import matplotlib.pyplot as plt
   from robustmodels.multivariate import RobustQDA

   X, y, names, _ = datasets.load_hemophilia()
   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.discriminant_plot(RobustLDA(random_state=0).fit(X, y), X, y, feature_names=names,
                           ax=axes[0])
   plots.discriminant_plot(RobustQDA(random_state=0).fit(X, y), X, y, feature_names=names,
                           ax=axes[1])
```

With a covariance per group, QDA bends the boundary but misclassifies as many
women as LDA here. The apparent error rate, computed on the training data, is
optimistic; estimate the error by cross-validation before relying on the rule.

Before classifying, a two-sample test confirms that the groups differ. The
permutation test compares the kernel density estimates of the two groups, and
the depth-depth plot shows how: most carriers are shallow in the normal group
and most normal women shallow among the carriers.

```pycon
>>> from robustmodels.density import kde_permutation_test
>>> normal, carrier = X[y == "normal"], X[y == "carrier"]
>>> kde_permutation_test(normal, carrier, random_state=0).pvalue
0.001

```

```{eval-rst}
.. plot::
   :filename-prefix: examples-hemophilia-ddplot
   :include-source: true
   :alt: Depth-depth plot of normal women and hemophilia A carriers.

   from robustmodels import datasets, plots

   X, y, *_ = datasets.load_hemophilia()
   ax = plots.dd_plot(X[y == "normal"], X[y == "carrier"], labels=["Normal", "Carrier"])
```

With 999 random assignments, 0.001 is the smallest attainable p-value: no
relabeling of the women gave a larger difference than the observed one.

### Robust logistic regression

The same question can be asked as a regression: how do the two measurements
change the odds of being a carrier? The Bianco–Yohai estimator bounds the
influence of misclassified or outlying women on the logistic fit. Carriers
have lower AHF activity for a given antigen level.

```pycon
>>> from robustmodels.glm import BYLogisticRegression
>>> carrier_indicator = (y == "carrier").astype(int)
>>> by = BYLogisticRegression().fit(X, carrier_indicator)
>>> print(by.summary(["AHFactivity", "AHFantigen"]))
               Estimate  Std. Error   z value  Pr(>|z|)
(Intercept)      -4.028       1.258    -3.203  0.00136
AHFactivity      -28.89       9.268    -3.117  0.00183
AHFantigen        29.01       11.67     2.485  0.013

```

`MallowsLogisticRegression` gives similar estimates here; see the
[generalized linear models guide](glm.md) for the differences between them.

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.datasets

.. autoclass:: Dataset
.. autofunction:: load_stars
.. autofunction:: load_hbk
.. autofunction:: load_wood
.. autofunction:: load_stackloss
.. autofunction:: load_hemophilia
```
