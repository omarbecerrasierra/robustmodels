# Depth and the bagplot

Depth measures centrality relative to a sample. `halfspace_depth` is the
smallest fraction of observations in a closed halfspace whose boundary passes
through the query point. It is exact in one and two dimensions, including ties.
Above two dimensions, it searches directions normal to sample hyperplanes and
returns an approximation that can overestimate the exact depth. Supply
`n_directions` and `random_state` to control that search.

`regression_depth` measures the smallest fraction of observations crossed when
tilting a fitted regression plane to a vertical nonfit. It is exact with one
predictor. With several predictors, enumerating all directions is exact for
data in general position; a limited random search can overestimate depth.
Predictors exclude the intercept column. Depths are fractions in `[0, 1]`.

## Bivariate center and outliers

`halfspace_median` returns the area centroid of the deepest bivariate region
(or the midpoint of a segment, or its only point). `bagplot_stats` builds the
bag containing approximately half the sample, its inflated fence, the outlier
mask, and the loop containing the bag and observations inside the fence.
`plots.bagplot` draws these statistics and returns a Matplotlib axes.

```pycon
>>> import numpy as np
>>> from robustmodels.depth import bagplot_stats, halfspace_depth
>>> square = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]
>>> halfspace_depth(square).tolist()
[0.2, 0.2, 0.2, 0.2, 0.6]
>>> X = np.random.default_rng(0).normal(size=(100, 2))
>>> X[:3] = [[6, 6], [7, -6], [-8, 5]]
>>> np.flatnonzero(bagplot_stats(X).outliers).tolist()
[0, 1, 2]

```

```{eval-rst}
.. plot::
   :filename-prefix: bagplot
   :include-source: true
   :alt: Bagplot with a Tukey median, central bag, fence and three outliers.

   import numpy as np
   from robustmodels import plots

   X = np.random.default_rng(0).normal(size=(100, 2))
   X[:3] = [[6, 6], [7, -6], [-8, 5]]
   plots.bagplot(X, feature_names=["First measurement", "Second measurement"])
```

## Limits and reference comparisons

The bivariate median and bag use a subset of at most `max_samples=200`
observations. Set a seed for larger samples. The complete data must span two
dimensions; a singular selected subset raises `ValueError` and requires a
larger `max_samples`. Duplicated observations are supported, including a bag
that collapses to a segment or a point. The bag boundary interpolates depth
regions over a finite angular grid, so transformed bag vertices need not be
exactly affine equivariant. Numerical ties use scale-relative tolerances.

Comparisons use mrfDepth and aplpack. Center conventions matter: mrfDepth's
`hdepthmedian` can choose a different representative of the deepest region
from the bagplot's centroid. On the car reference, mrfDepth's center has depth
24/60 and the exact halfplane intersection here yields depth 25/60. That is
checked independently rather than enforcing numerical equality with R.
The captured mrfDepth 1.0.17 three-predictor regression-depth result was
outside `[0, 1]`; it is excluded as a numerical witness. Small problems in
three predictor dimensions are instead checked by exhaustive deletion and
linear-programming separation.

Sources are Tukey (1975), Rousseeuw and Ruts (1996, 1998), Rousseeuw and Hubert
(1999), and Rousseeuw, Ruts and Tukey (1999); full citations follow.

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.depth

.. autofunction:: halfspace_depth
.. autofunction:: regression_depth
.. autofunction:: halfspace_median
.. autofunction:: bagplot_stats
```
