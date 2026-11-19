# Robust clustering

Clustering by trimming, in `robustmodels.cluster`. A few outliers can pull a
k-means center or a normal mixture component towards them, or take a cluster
of their own. `TClust` discards the fraction `alpha` of observations that no
cluster explains well, and fits normal clusters to the rest.

## Use

```pycon
>>> import numpy as np
>>> from robustmodels.cluster import TClust
>>> rng = np.random.default_rng(1)
>>> X = np.vstack(
...     [
...         rng.multivariate_normal([0, 0], np.eye(2), 100),
...         rng.multivariate_normal([5, 5], [[4, 0], [0, 0.3]], 80),
...         rng.multivariate_normal([-4, 6], [[1, 0.8], [0.8, 1]], 60),
...         rng.uniform(-10, 12, (12, 2)),  # scattered noise
...     ]
... )
>>> fit = TClust(n_clusters=3, alpha=0.05, random_state=0).fit(X)
>>> np.bincount(fit.labels_ + 1).tolist()  # trimmed, then each cluster
[13, 99, 79, 61]
>>> fit.means_.round(2).tolist()
[[-0.05, -0.05], [4.52, 4.95], [-4.16, 5.95]]

```

`labels_` is `-1` for trimmed observations, and the clusters are ordered by
size. `means_`, `covariances_` and `weights_` describe the fitted normal
clusters; `predict(X)` assigns new observations and trims those less plausible
than every retained training observation.

```{eval-rst}
.. plot::
   :filename-prefix: tclust
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
   plots.cluster_plot(TClust(3, random_state=0).fit(X), X)
```

## Choosing the parameters

- **`restr_factor`** bounds the ratio between the largest and the smallest
  eigenvalue of all the cluster covariances. Without a bound the likelihood
  is unbounded (a cluster can collapse on a few points). The default 12, as in
  R, allows elongated clusters of different sizes; `1` forces spherical
  clusters of equal spread.
- **`equal_weights=True`** assumes clusters of similar size. With
  `restr_factor=1` it gives trimmed k-means.
- **`alpha`** should exceed the expected share of outliers; trimming a few
  regular observations costs little. Fit several values of `alpha` and
  `n_clusters` and compare the solutions and their plots; the trimmed
  likelihood (`objective_`) always grows with the number of clusters, so do
  not choose `n_clusters` by maximizing it.
- **`n_init`** random starts (500 by default) are improved by concentration
  steps; the search returns a local maximum, so keep many starts.

The classification trimmed likelihood curves (García-Escudero et al. 2011),
R's `ctlcurves`, guide both choices: they draw `objective_` against `alpha`
for each number of clusters. Choose the smallest `k` whose curve is close to
that of `k + 1`, and the smallest `alpha` from which they stay close. Here the
curves for 3 and 4 clusters merge from `alpha = 0.04`: three clusters, with
about 5% of noise.

```{eval-rst}
.. plot::
   :filename-prefix: ctlcurves
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
   plots.ctl_curves(X, n_init=100, random_state=0)
```

## Numerical checks and sources

The eigenvalue restriction is the optimal truncation of Fritz,
García-Escudero and Mayo-Iscar (2013); the tests compare it with a
brute-force minimization of its criterion over a fine grid. The fitted model
reproduces its own assignment and trimming of the training data, and trimmed
k-means assigns each retained observation to its nearest center and trims the
farthest ones. On frozen data, R's tclust 2.2 finds the same partition, means,
weights and objective (to 1e-10), and tkmeans the same partition and centers.
The starts, concentration steps and constants follow tclust's code.

Sources are García-Escudero, Gordaliza, Matrán and Mayo-Iscar (2008), Fritz,
García-Escudero and Mayo-Iscar (2012, 2013) and Cuesta-Albertos, Gordaliza and
Matrán (1997).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.cluster

.. autoclass:: TClust
   :members: fit, predict
```
