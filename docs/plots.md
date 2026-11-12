# Diagnostic plots for robust models

The five functions in `robustmodels.plots` turn fitted estimates into
Matplotlib diagnostics. Each returns an `Axes`, accepts an existing `ax`,
and styles that axes without changing global `rcParams`.

## Choosing a plot

| Function | Input | What to inspect |
| --- | --- | --- |
| `outlier_map(model, X, y)` | A fitted regressor with `predict` and `scale_` | Vertical outliers, good leverage and bad leverage |
| `distance_distance(model, X)` | A fitted covariance estimator | Observations masked by the classical covariance |
| `tolerance_ellipse(model, X)` | A fitted covariance estimator for exactly two variables | Robust and classical centers, spread and flagged rows |
| `adjusted_boxplot(x)` | A finite, nonempty vector or matrix | Marginal outliers after allowing for skewness |
| `outlyingness_plot(detector, X=None)` | A fitted projection detector | Scores and flags for training or new observations |

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

The [validation notes](validation.md) record the pre-release comparison and
its limits. These plots are tools for investigating observations, not rules
for automatically deleting them.

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.plots

.. autofunction:: outlier_map
.. autofunction:: distance_distance
.. autofunction:: tolerance_ellipse
.. autofunction:: adjusted_boxplot
.. autofunction:: outlyingness_plot
```
