# Diagnostic plots for robust models

The functions in `robustmodels.plots` turn fitted estimates into
Matplotlib diagnostics. Each returns an `Axes`, accepts an existing `ax`,
and styles that axes without changing global `rcParams`.

## Choosing a plot

| Function | Input | What to inspect |
| --- | --- | --- |
| `outlier_map(model, X, y)` | A fitted regressor with `predict` and `scale_` | Vertical outliers, good leverage and bad leverage |
| `distance_distance(model, X)` | A fitted covariance estimator | Observations masked by the classical covariance |
| `tolerance_ellipse(model, X)` | A fitted covariance estimator for exactly two variables | Robust and classical centers, spread and flagged rows |
| `adjusted_boxplot(x)` | A finite, nonempty vector or matrix | Marginal outliers after allowing for skewness |
| `bagplot(X)` | A nonsingular two-column sample | Bivariate center, spread, skewness and outliers |
| `outlyingness_plot(detector, X=None)` | A fitted projection detector | Scores and flags for training or new observations |
| `residual_plot(model, X, y)` | A fitted regressor with `predict` and `scale_` | Outlying residuals, trends and changing spread along the fitted values |
| `residual_qq(model, X, y)` | A fitted regressor with `predict` and `scale_` | Whether the regular residuals look normal, and which do not |
| `distance_qq(model, X)` | A fitted covariance estimator | Whether the regular robust distances follow the chi-square reference |
| `cluster_plot(model, X)` | A fitted `TClust` | Clusters, their 97.5% ellipses and the trimmed observations |
| `hdr_plot(model)` | A fitted `KernelDensity` of one or two variables | Highest density regions and modes |
| `log_concave_plot(model)` | A fitted `LogConcaveDensity` | The estimate and its knots against the histogram, or the log density |
| `score_plot(model, X, y)` | A fitted `ScoreMatchingRegressor` | The estimated score and how it bounds large residuals |
| `coefficient_plot(models, X=, y=)` | Fitted regressors or GLMs with `std_errors_` | Whether robust and least squares fits lead to different conclusions |
| `dd_plot(X1, X2)` | Two samples with the same variables | Whether two samples share a distribution, and how they differ |
| `ctl_curves(X)` | A sample to cluster | How many clusters and how much trimming |
| `pca_outlier_map(model, X=None)` | A fitted `ROBPCA` or `ProjectionPursuitPCA` | Orthogonal outliers, good and bad leverage points of a PCA |
| `scree_plot(model)` | A fitted robust PCA | How many components carry the variance |
| `biplot(model, X)` | A fitted robust PCA with at least two components | Scores of the observations and loadings of the variables together |
| `discriminant_plot(model, X, y)` | A fitted `RobustLDA` or `RobustQDA` for exactly two variables | Class regions, boundaries and misclassified observations |

Rows are observations and columns are variables. Fit the estimator first;
`outlier_map` additionally fits an MCD to the supplied predictors, while the
other model-based functions use the supplied fit. Pass the same predictor
columns, in the same order, as when fitting the model.

Figures below are generated from their displayed code when the documentation
is built. The examples use fixed random seeds and require no downloaded data.

## Regression outlier map

The vertical coordinate is the residual divided by the fitted residual scale,
`(y - model.predict(X)) / model.scale_`. The horizontal coordinate is the
**unsquared** MCD distance of the predictors. The thresholds are `±2.5` and
`sqrt(chi2.ppf(0.975, p))`, where `p` counts predictors, **excluding the
intercept**.

| Class | Robust predictor distance | Absolute standardized residual |
| --- | --- | --- |
| Regular | At or below the distance cutoff | At or below 2.5 |
| Vertical outlier | At or below the distance cutoff | Above 2.5 |
| Good leverage | Above the distance cutoff | At or below 2.5 |
| Bad leverage | Above the distance cutoff | Above 2.5 |

```{eval-rst}
.. plot::
   :filename-prefix: outlier-map
   :include-source: true
   :alt: Regression diagnostic separating vertical outliers and good and bad leverage.

   import numpy as np
   from robustmodels import plots
   from robustmodels.regression import MMRegressor

   rng = np.random.default_rng(0)
   X = rng.normal(size=(100, 2))
   X[:10] += 6
   y = 1 + X @ [1.0, 2.0] + rng.normal(scale=0.5, size=100)
   y[:5] += 12          # bad leverage: extreme predictors and response off the line
   y[10:15] += 12      # vertical outliers; rows 5:10 remain good leverage
   model = MMRegressor(random_state=0).fit(X, y)
   ax = plots.outlier_map(model, X, y, random_state=0)
```

`random_state` controls the additional MCD search, independently of the
regression fit. A zero residual scale is an exact fit, for which standardized
residuals are undefined: the function raises `ValueError`. The residual
cutoff is a diagnostic convention; it differs from the approximately 2.24
threshold used internally to reweight LTS regression.

By default, up to three eligible observations are labeled with their
**zero-based row positions**, not DataFrame index labels. In this plot their
severity is the larger of `abs(residual) / 2.5` and `distance / distance_cutoff`.
A good leverage point can therefore be labeled ahead of a bad leverage point.
Set the positive integer `id_n` to change the maximum number of labels.

## Distance-distance plot

```{eval-rst}
.. plot::
   :filename-prefix: distance-distance
   :include-source: true
   :alt: Robust versus classical Mahalanobis distances for a sample with ten shifted rows.

   import numpy as np
   from robustmodels import plots
   from robustmodels.covariance import MCD

   X = np.random.default_rng(0).normal(size=(100, 2))
   X[:10] += 10
   model = MCD(random_state=0).fit(X)
   ax = plots.distance_distance(model, X)
```

Both axes show unsquared distances; the estimator's `mahalanobis` method
returns squared distances and the plot takes their square root. The classical
fit is recomputed from `X` using its mean and sample covariance (divisor `n−1`).
The robust fit is the supplied model. A row is flagged when its robust
distance exceeds the 97.5% chi-square reference cutoff.

A point with a large robust distance but a small classical distance is a
candidate for masking: the classical fit has absorbed the contamination.
Being slightly above the diagonal alone is not an outlier rule. The labels
identify the largest flagged robust distances.

## Tolerance ellipses

```{eval-rst}
.. plot::
   :filename-prefix: tolerance-ellipse
   :include-source: true
   :alt: Robust and classical 97.5 percent reference ellipses with flagged observations.

   import numpy as np
   from robustmodels import plots
   from robustmodels.covariance import MCD

   X = np.random.default_rng(4).multivariate_normal(
       [0, 0], [[1, 0.7], [0.7, 1]], size=100
   )
   X[:8] += [5, -3]
   model = MCD(random_state=0).fit(X)
   ax = plots.tolerance_ellipse(model, X, feature_names=["Measurement 1", "Measurement 2"])
```

Each boundary satisfies `(x - center)' covariance^-1 (x - center) =
chi2.ppf(0.975, 2)`. The robust boundary uses `model.location_` and
`model.covariance_`; the classical boundary uses the sample mean and
covariance with divisor `n−1`. Crosses mark the centers. Flags and labels use
robust distances, so a flagged observation is outside the robust ellipse.

Pass data with exactly two columns and, if supplied, exactly two
`feature_names`. Fit a two-dimensional model to those columns first; selecting
two columns from a higher-dimensional fit does not define this diagnostic.

The 97.5% level is a multivariate-normal **reference contour**. It is not a
confidence region for the estimated center, an exact finite-sample tolerance
region or a guarantee that 2.5% of a fitted sample will be flagged. Distances
and chi-square calibration require a nonsingular covariance; a pseudoinverse
on a degenerate sample does not restore that calibration.

## Skewness-adjusted boxplot

```{eval-rst}
.. plot::
   :filename-prefix: adjusted-boxplot
   :include-source: true
   :alt: Adjusted boxplots for right-skewed, symmetric and left-skewed samples.

   import numpy as np
   from robustmodels import plots

   rng = np.random.default_rng(0)
   X = np.column_stack([
       rng.lognormal(size=200),
       rng.normal(size=200),
       -rng.exponential(size=200),
   ])
   ax = plots.adjusted_boxplot(X, labels=["lognormal", "normal", "-exponential"])
```

One box is drawn per column. The box uses Tukey's lower and upper hinges, as
R's `fivenum`, and the middle line is the median. Let `IQR = Q3 - Q1` and `MC`
be the medcouple. The lower and upper fences are:

| Medcouple | Lower fence | Upper fence |
| --- | --- | --- |
| `MC >= 0` | `Q1 - 1.5 exp(-4 MC) IQR` | `Q3 + 1.5 exp(3 MC) IQR` |
| `MC < 0` | `Q1 - 1.5 exp(-3 MC) IQR` | `Q3 + 1.5 exp(4 MC) IQR` |

Whiskers end at the most extreme **observed values inside the fences**, not
at the fences themselves. Values strictly beyond a fence are drawn as orange
diamonds; values on a fence stay inside. For `MC = 0` the usual 1.5-IQR fences
are recovered. Positive skewness lengthens the upper fence and shortens the
lower one; reflecting the sample exchanges the two sides.

The medcouple above each box summarizes its skewness. With several columns,
the subtitle counts flagged values across columns, so a row flagged in two
variables contributes twice. This is a marginal diagnostic and does not
detect every multivariate outlier. Supply one label per column; empty,
nonfinite or higher-dimensional input raises `ValueError`.

These hinges follow `robustbase::adjboxStats`; the projection detector's
adjusted outlyingness instead uses type-5 quartiles, following mrfDepth. Both
use this package's medcouple definition, including its documented handling
of ties, which can differ from R; see [the method references](references.md#bhs04).

## Outlyingness by observation

```{eval-rst}
.. plot::
   :filename-prefix: outlyingness
   :include-source: true
   :alt: Adjusted outlyingness scores with the fitted cutoff and five flagged observations.

   import numpy as np
   from robustmodels import plots
   from robustmodels.outliers import AdjustedOutlyingness

   X = np.random.default_rng(0).lognormal(size=(100, 2))
   X[:5] = [-3, -3]
   detector = AdjustedOutlyingness(random_state=0).fit(X)
   ax = plots.outlyingness_plot(detector)
```

Without `X`, the plot uses `detector.outlyingness_`. With `X`, it evaluates
`detector.outlyingness(X)` against the **existing fit**, and keeps its training
`cutoff_`; it does not refit or recalibrate on the new observations.

The horizontal coordinate is the zero-based position in the plotted data.
Finite scores above the cutoff are orange diamonds. Positive infinite
outlyingness, possible when a projection has zero scale, is displayed as an
orange triangle at a finite height above the finite scores and cutoff. That
height is only a plotting position, not a replacement for the infinite score.

The legend gives separate counts for regular observations, finite flagged
scores and infinite scores; empty classes are omitted. The subtitle counts
all scores above the cutoff, including positive infinities. Up to `id_n`
flagged observations with the largest scores are labeled, keeping input order
when scores tie. Nearby labels are separated with small offsets and, when
needed, leader lines.

`StahelDonohoOutlyingness` can be used in the same way. The plot displays the
detector's rule; it does not introduce a second outlier criterion. See
[outlier detection](outliers.md) for the log-scale cutoff and approximation
by a finite set of projection directions.

## PCA outlier map

```{eval-rst}
.. plot::
   :filename-prefix: pca-outlier-map
   :include-source: true
   :alt: PCA outlier map with orthogonal outliers, a good leverage point and bad leverage points.

   import numpy as np
   from robustmodels import plots
   from robustmodels.multivariate import ROBPCA

   rng = np.random.default_rng(0)
   X = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 6))
   X += 0.3 * rng.normal(size=X.shape)
   X[:8] += 4
   ax = plots.pca_outlier_map(ROBPCA(2, random_state=0).fit(X))
```

The horizontal axis is the score distance, the robust distance of an
observation's scores from the center inside the principal subspace. The
vertical axis is the orthogonal distance from the observation to that
subspace. The cutoffs are those of the fitted model: the square root of the
0.975 chi-square quantile with as many degrees of freedom as components, and
`orthogonal_cutoff_`.

| Region | Class | Reading |
| --- | --- | --- |
| Below, left | Regular | Close to the subspace and to the center |
| Above, left | Orthogonal outliers | Far from the subspace; their scores look ordinary |
| Below, right | Good leverage | In the subspace but far from the center |
| Above, right | Bad leverage | Far from the subspace and from the center |

Without `X` the map shows the training data; with `X` it places new
observations against the existing fit and cutoffs. A model that keeps as many
components as the rank of the data has no orthogonal distances, and the
function raises `ValueError`. See the
[PCA guide](multivariate.md#principal-components).

## Scree plot and biplot

```{eval-rst}
.. plot::
   :filename-prefix: scree-biplot
   :include-source: true
   :alt: Scree plot of five robust components beside a biplot of the first two.

   import matplotlib.pyplot as plt
   import numpy as np
   from robustmodels import plots
   from robustmodels.multivariate import ROBPCA

   rng = np.random.default_rng(0)
   X = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 6))
   X += 0.3 * rng.normal(size=X.shape)
   X[:8] += 4
   names = ["Ca", "Mg", "Na", "K", "Fe", "Zn"]

   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.scree_plot(ROBPCA(5, random_state=0).fit(X), ax=axes[0])
   plots.biplot(ROBPCA(2, random_state=0).fit(X), X, feature_names=names, ax=axes[1])
```

The scree plot shows the robust variance of each fitted component; the label
above each point is the cumulative share of the variance of the fitted
components, not of the total variance of the data, which a robust fit with
fewer components than variables does not estimate. Fit as many components as
you want to compare, then refit with the number you keep.

In the biplot, points are scores on two components (`components=(0, 1)` by
default) and arrows are the loadings of the variables on the same components.
All arrows are multiplied by one factor, so only their directions and relative
lengths are meaningful: variables with arrows in the same direction vary
together in this plane. Observations beyond the score or the orthogonal cutoff
of the model are flagged, so an orthogonal outlier can appear in the middle of
the cloud. `n_arrows` limits the arrows to the longest ones.

## Classification regions

```{eval-rst}
.. plot::
   :filename-prefix: discriminant
   :include-source: true
   :alt: Quadratic classification regions of three classes with misclassified points circled.

   import numpy as np
   from robustmodels import plots
   from robustmodels.multivariate import RobustQDA

   rng = np.random.default_rng(0)
   X = np.vstack([
       rng.normal(size=(50, 2)),
       2 * rng.normal(size=(50, 2)) + 4,
       rng.normal(size=(50, 2)) + [5, -2],
   ])
   y = np.repeat(["a", "b", "c"], 50)
   X[:5] = [9, 9]  # five outliers labeled "a"
   ax = plots.discriminant_plot(RobustQDA(random_state=0).fit(X, y), X, y)
```

Shaded regions show the class predicted at each point of the plane and gray
lines the boundaries. Crosses and colored ellipses are the robust center and
97.5% tolerance ellipse of each class, the display used by
[Hubert and Van Driessen](references.md#hvd04); `RobustLDA` draws the same
shape for every class. Observations keep the color
and marker of their true class; those assigned to another class are circled.
The subtitle reports the share of plotted observations that are misclassified.
For the training data this apparent error is optimistic: report the error on
held-out data or by cross-validation. The function needs exactly two
variables.

## Residual and quantile plots

```{eval-rst}
.. plot::
   :filename-prefix: residual-quantile-plots
   :include-source: true
   :alt: Residuals against fitted values, their normal quantile plot and a chi-square quantile plot of robust distances.

   import matplotlib.pyplot as plt
   import numpy as np
   from robustmodels import plots
   from robustmodels.covariance import MCD
   from robustmodels.regression import MMRegressor

   rng = np.random.default_rng(0)
   X = rng.normal(size=(100, 3))
   y = 1 + X @ [2.0, -1.0, 0.5] + rng.normal(scale=0.5, size=100)
   X[:10] += 4
   y[10:20] += 6
   model = MMRegressor(random_state=0).fit(X, y)

   fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), layout="constrained")
   plots.residual_plot(model, X, y, ax=axes[0])
   plots.residual_qq(model, X, y, ax=axes[1])
   plots.distance_qq(MCD(random_state=0).fit(X), X, ax=axes[2])
```

`residual_plot` shows the standardized residuals against the fitted values,
with +-2.5 as cutoffs. A trend or a funnel shape among the regular observations
points to a missing term or to a variance that changes with the mean.

`distance_qq` plots the sorted robust distances against the square roots of
chi-square quantiles, and `residual_qq` the sorted standardized residuals
against normal quantiles, both at probabilities `(i - 0.5) / n`. Regular
observations from the normal model follow the diagonal. Outliers leave it at
the ends and lie beyond the dashed cutoffs: the square root of the 0.975
chi-square quantile, and +-2.5. A curved pattern in the regular observations
suggests that the normal reference does not describe the bulk of the data, in
which case the cutoffs are only rough guides. The distances and residuals come
from the robust fit, so the outliers do not bend the reference themselves.

## Bagplot

```{eval-rst}
.. plot::
   :filename-prefix: plots-bagplot
   :include-source: true
   :alt: Bagplot with a Tukey median, central bag, fence and three outliers.

   import numpy as np
   from robustmodels import plots

   X = np.random.default_rng(0).normal(size=(100, 2))
   X[:3] = [[6, 6], [7, -6], [-8, 5]]
   ax = plots.bagplot(X, feature_names=["First measurement", "Second measurement"])
```

The bagplot is the bivariate boxplot. The cross is the Tukey median, the dark
bag holds the half of the observations with the largest halfspace depth, and
the dashed fence is the bag inflated by `factor` (3 by default) from the
median. The light loop is the convex hull of the observations inside the
fence; observations outside it are flagged. A bag elongated along a diagonal
shows correlation, and a bag off center inside the loop shows skewness. For
more than 200 observations the median and the bag are computed on a random
subset of `max_samples` rows, controlled by `random_state`. See the
[depth guide](depth.md).

## Clusters and trimmed observations

```{eval-rst}
.. plot::
   :filename-prefix: plots-cluster
   :include-source: true
   :alt: Three clusters of different shapes with 97.5% ellipses and trimmed noise points.

   import numpy as np
   from robustmodels import plots
   from robustmodels.cluster import TClust

   rng = np.random.default_rng(1)
   X = np.vstack([
       rng.multivariate_normal([0, 0], np.eye(2), 100),
       rng.multivariate_normal([5, 5], [[4, 0], [0, 0.3]], 80),
       rng.multivariate_normal([-4, 6], [[1, 0.8], [0.8, 1]], 60),
       rng.uniform(-10, 12, (12, 2)),
   ])
   ax = plots.cluster_plot(TClust(3, random_state=0).fit(X), X)
```

Observations are colored by cluster and the trimmed ones are orange
diamonds. Crosses and ellipses are the center and 97.5% tolerance ellipse of
each cluster's normal model. With more than two variables, `features` selects
the two columns on the axes and the ellipses are the marginal ones of those
columns. Passing other data than the training sample assigns it with
`model.predict`. See the [clustering guide](cluster.md).

## Choosing the number of clusters and the trimming

```{eval-rst}
.. plot::
   :filename-prefix: plots-ctlcurves
   :include-source: true
   :alt: Trimmed likelihood against the trimming level for one to four clusters.

   import numpy as np
   from robustmodels import plots

   rng = np.random.default_rng(4)
   X = np.vstack([
       rng.multivariate_normal([0, 0], np.eye(2), 100),
       rng.multivariate_normal([5, 5], [[4, 0], [0, 0.3]], 80),
       rng.multivariate_normal([-4, 6], [[1, 0.8], [0.8, 1]], 60),
       rng.uniform(-10, 12, (15, 2)),
   ])
   ax = plots.ctl_curves(X, n_init=100, random_state=0)
```

`ctl_curves` fits `TClust` for every number of clusters in `n_clusters` and
every trimming level in `alphas`, and draws the trimmed classification
log-likelihood against `alpha`, one curve per number of clusters. The
likelihood always grows with the number of clusters. Choose the smallest `k`
whose curve is close to that of `k + 1`, and the smallest `alpha` from which
they stay close. Here the curves for three and four clusters merge from
`alpha = 0.04`. Each point is a full fit, so the plot takes
`len(n_clusters) * len(alphas)` times as long as one fit; lower `n_init` for
a first look.

## Highest density regions

```{eval-rst}
.. plot::
   :filename-prefix: plots-hdr
   :include-source: true
   :alt: Filled 25%, 50% and 75% highest density regions of a two-mode sample.

   import numpy as np
   from robustmodels import plots
   from robustmodels.density import KernelDensity

   rng = np.random.default_rng(1)
   X = np.vstack([
       rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 250),
       rng.multivariate_normal([3, 3], [[0.4, -0.2], [-0.2, 0.4]], 150),
   ])
   ax = plots.hdr_plot(KernelDensity().fit(X))
```

The region of probability `prob` is the smallest region with that
probability, `{x : f(x) >= level}`, with the level from
`model.hdr_level(prob)`. For two variables the regions are filled, darker for
the smaller ones, over the observations; for one variable they are shaded
under the density curve over a rug of the observations. A region that splits
into separate pieces reveals several modes; here every region has one piece
around each of the two modes. See
the [density guide](density.md).

## Log-concave density

```{eval-rst}
.. plot::
   :filename-prefix: plots-logconcave
   :include-source: true
   :alt: Log-concave density estimate and its log density for a gamma sample.

   import matplotlib.pyplot as plt
   import numpy as np
   from robustmodels import plots
   from robustmodels.shape import LogConcaveDensity

   x = np.random.default_rng(3).gamma(2, size=250)
   model = LogConcaveDensity().fit(x)

   fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
   plots.log_concave_plot(model, ax=axes[0])
   plots.log_concave_plot(model, log=True, ax=axes[1])
```

The left panel draws the estimate over the histogram of the sample, and the
right panel, with `log=True`, its logarithm. The log density is piecewise
linear and concave; the knots are the observations where its slope changes.
The estimate is zero outside the range of the sample. A histogram with two
modes or heavier tails than the estimate contradicts the log-concave
assumption. See the [shape-constrained guide](shape.md).

## Antitonic score

```{eval-rst}
.. plot::
   :filename-prefix: plots-score
   :include-source: true
   :alt: Estimated antitonic score, bounded in the tails, against the normal score.

   import numpy as np
   from robustmodels import plots
   from robustmodels.shape import ScoreMatchingRegressor

   rng = np.random.default_rng(4)
   X = rng.normal(size=(400, 2))
   y = 1 + X @ [2.0, -1.0] + rng.standard_t(2, 400)
   ax = plots.score_plot(ScoreMatchingRegressor().fit(X, y), X, y)
```

The curve is the estimated score `psi`, minus the derivative of the convex
loss, against the residual, over a rug of the residuals. The straight line is
the score of a normal distribution with the MAD of the residuals, which least
squares uses. Where the estimated score flattens, large residuals stop gaining
influence: here the errors have Student t tails with two degrees of freedom.
The horizontal axis covers the central 98% of the residuals, widened by a
quarter of that range on each side. See the
[shape-constrained guide](shape.md#regression-by-antitonic-score-matching).

## Comparing coefficients across fits

```{eval-rst}
.. plot::
   :filename-prefix: plots-coefficients
   :include-source: true
   :alt: Coefficient estimates with 95% intervals from MM, rank and score matching fits and least squares.

   import numpy as np
   from robustmodels import plots
   from robustmodels.rank import RankRegressor
   from robustmodels.regression import MMRegressor
   from robustmodels.shape import ScoreMatchingRegressor

   rng = np.random.default_rng(4)
   X = rng.normal(size=(150, 3))
   y = 1 + X @ [2.0, -1.0, 0.0] + rng.standard_t(2, 150)
   y[:8] += 25 * X[:8, 2]  # eight observations tie y to the third variable
   fits = [MMRegressor().fit(X, y), RankRegressor().fit(X, y), ScoreMatchingRegressor().fit(X, y)]
   ax = plots.coefficient_plot(fits, labels=["MM", "Rank", "Score matching"], X=X, y=y,
                               feature_names=["dose", "age", "site"])
```

Each row is a coefficient and each fit a marker with its interval
`estimate ± z * standard error`, with `z` the normal quantile of `level`
(0.95 by default). Passing both `X` and `y` adds least squares with its
classical standard errors. Here eight observations make the third variable
look important to least squares, while the robust fits agree that it is not.
Every model needs `std_errors_` and the same predictors; set
`intercept=False` to hide the intercept row. See the
[regression guide](regression.md#comparing-fits).

## Depth-depth plot

```{eval-rst}
.. plot::
   :filename-prefix: plots-ddplot
   :include-source: true
   :alt: Depth-depth plot of two samples whose spreads differ, with points off the diagonal.

   import numpy as np
   from robustmodels import plots

   rng = np.random.default_rng(4)
   control = rng.normal(size=(150, 2))
   treated = rng.normal(size=(120, 2)) * [1.6, 1]
   ax = plots.dd_plot(control, treated, labels=["Control", "Treated"])
```

Every observation of both samples is placed at its halfspace depth in the
first sample and in the second. Samples from one distribution keep the points
near the diagonal. A difference in location bends them into a lens, and a
difference in spread, as here, moves them to one side. For three or more
variables the depth is approximated with random directions controlled by
`random_state`. The plot is descriptive; `density.kde_permutation_test` and
`density.kde_test` test the difference. See the
[depth guide](depth.md#comparing-two-samples).

## Customizing and saving

Use `ax=` to compose several diagnostics. Functions return the supplied axes,
so Matplotlib methods remain available after plotting:

```pycon
>>> import matplotlib.pyplot as plt
>>> from robustmodels import plots
>>> fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
>>> left = plots.adjusted_boxplot([1, 2, 3, 4, 9], ax=axes[0])
>>> right = plots.adjusted_boxplot([-9, -4, -3, -2, -1], ax=axes[1])
>>> left is axes[0] and right is axes[1]
True
>>> plt.close(fig)

```

Save the returned figure with `ax.figure.savefig("diagnostic.png", dpi=300)`
or `ax.figure.savefig("diagnostic.svg")` for vector output. In a script, use
`plt.show()` to display it, and `plt.close(ax.figure)` after saving many
figures. On a machine without a display, select Matplotlib's `Agg` backend
before importing `matplotlib.pyplot`.

Colors and marker shapes identify groups consistently. Empty scatter groups
are omitted from legends. The style is applied locally; the functions do not
change Matplotlib's global style. Very large `id_n` values can still crowd a
figure; inspect its final size before publication.

## Sources and R conventions

- The regression map follows [Rousseeuw and van Zomeren](references.md#rvz90).
  The chi-square degrees of freedom count predictors; `plot.lmrob` in
  robustbase 0.99-7 uses the regression rank, including the intercept.
- Classical distances use the sample covariance as `robustbase::covPlot`.
  The robust distances come directly from the fitted estimator.
- Ellipses use the supplied covariance without an additional multiplier.
  `robustbase::tolEllipsePlot` in 0.99-7 multiplies both covariance matrices
  by `n / (n - 1)` when drawing its contours. We omit that extra inflation so
  the robust boundary agrees with the distance used to flag observations.
- The adjusted fences follow [Hubert and Vandervieren](references.md#hv08).
  Hinges and whiskers follow `adjboxStats`; medcouple approximation and ties
  can produce numerical differences from R without changing the definition.

- The PCA outlier map follows [Hubert, Rousseeuw and Vanden Branden](references.md#hrvb05),
  with their univariate-MCD cutoff for the orthogonal distances;
  `rrcov`'s `plot` method for `PcaHubert` uses a median and MAD cutoff.
- `distance_distance`, `tolerance_ellipse` and `outlier_map` raise `ValueError`
  when the robust covariance is singular. A pseudo-inverse gives distance
  zero along the directions without spread, so points far away in those
  directions would sit inside a contour labeled 97.5%.

## Figures for publication

Every plot is sized for one journal column (5.2 × 3.8 inches), uses one color
per role across the plots, and pairs each color with its own marker so that
figures remain readable in grayscale and for color-blind readers. Save vector
figures with `ax.figure.savefig("figure.pdf")`. Many submission systems
(IEEE, Elsevier) reject the Type 3 fonts that Matplotlib embeds in PDF files
by default; select TrueType fonts before plotting:

```python
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
```

Combine several plots in one figure by passing each its own axes, for example
from `plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")`.

The [validation notes](validation.md) record the pre-release comparison and
its limits. These plots are tools for investigating observations, not rules
for automatically deleting them. The [examples with classic data](examples.md)
apply them to data sets whose outliers are documented.

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.plots

.. autofunction:: outlier_map
.. autofunction:: distance_distance
.. autofunction:: tolerance_ellipse
.. autofunction:: adjusted_boxplot
.. autofunction:: outlyingness_plot
.. autofunction:: residual_plot
.. autofunction:: residual_qq
.. autofunction:: distance_qq
.. autofunction:: pca_outlier_map
.. autofunction:: scree_plot
.. autofunction:: bagplot
.. autofunction:: biplot
.. autofunction:: discriminant_plot
.. autofunction:: cluster_plot
.. autofunction:: hdr_plot
.. autofunction:: log_concave_plot
.. autofunction:: score_plot
.. autofunction:: coefficient_plot
.. autofunction:: dd_plot
.. autofunction:: ctl_curves
```
