# Robust PCA, discriminant analysis, Hotelling's T² and MANOVA

Multivariate methods that keep working when part of the observations are
outliers, in `robustmodels.multivariate`: principal components, linear and
quadratic discriminant analysis, and tests on mean vectors.

## Choosing a method

| Task | Class or function | Notes |
| --- | --- | --- |
| Principal components | `ROBPCA` | The standard choice; works with more variables than observations |
| Principal components without a covariance matrix | `ProjectionPursuitPCA` | Maximizes a robust scale direction by direction; more biased by outliers near the data |
| Classify with a common covariance | `RobustLDA` | Linear boundaries; pooled MCD |
| Classify with one covariance per class | `RobustQDA` | Quadratic boundaries; one MCD per class |
| Test a mean vector, or paired differences | `hotelling_t2` | One sample; reweighted MCD |
| Compare the means of two or more groups | `manova` | Robust Wilks' lambda; simulated null distribution |

The discriminant rules and the tests are built on
[`covariance.MCD`](covariance.md), so every group needs more observations than
variables. For a robust covariance matrix itself, including the Stahel-Donoho
estimator, see the [covariance guide](covariance.md).

## Principal components

Classical PCA follows the directions of largest variance, and a few outliers
are enough to create one. ROBPCA finds the subspace of the bulk of the data:

```pycon
>>> import numpy as np
>>> from robustmodels.multivariate import ROBPCA
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 20))  # a plane in 20-D
>>> X += 0.1 * rng.normal(size=X.shape)
>>> X[:10] += 5  # ten outliers away from the plane
>>> pca = ROBPCA(random_state=0).fit(X)
>>> pca.n_components_, pca.explained_variance_.round(1).tolist()
(2, [42.6, 24.2])
>>> pca.transform(X).shape
(100, 2)
>>> np.flatnonzero(~pca.support_).tolist()
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 69, 70]

```

The ten outliers are flagged, with three regular observations in the tails. The
largest angle between the fitted plane and the plane of the 90 regular
observations is 0.1 degrees; for classical PCA of all rows it is 45 degrees.

| Attribute or method | Meaning |
| --- | --- |
| `location_`, `components_`, `explained_variance_` | Robust center, principal axes in rows and variance of each score |
| `transform(X)`, `inverse_transform(scores)` | Scores, and their points in the original space |
| `distances(X)` | Score and orthogonal distances of new observations |
| `score_distances_`, `score_cutoff_` | Distance inside the subspace and its 0.975 chi-square cutoff |
| `orthogonal_distances_`, `orthogonal_cutoff_` | Distance to the subspace and its cutoff |
| `support_` (`ROBPCA`) | Observations within both cutoffs |

An observation can be outlying in two ways: far from the subspace (a large
orthogonal distance) or far from the center inside it (a large score distance).
[`plots.pca_outlier_map`](plots.md#pca-outlier-map) shows both;
[`plots.scree_plot`](plots.md#scree-plot-and-biplot) and `plots.biplot` show
the variances, scores and loadings.

With `n_components=None`, ROBPCA keeps the smallest number of components that
explains 80% of the variance of the least outlying observations, at most
`max_components`. Pass `n_components` to choose it yourself, for example after
looking at `explained_variance_` of a fit with more components.

`ProjectionPursuitPCA` needs no covariance estimate. It searches, one after
another, the orthogonal directions on which a robust scale of the projected
data is largest:

```pycon
>>> from robustmodels.multivariate import ProjectionPursuitPCA
>>> X = np.random.default_rng(0).normal(size=(100, 3)) * [5, 1, 1]
>>> X[:10] = [0, 100, 0]  # ten outliers on the second axis
>>> abs(np.linalg.svd(X - X.mean(axis=0))[2][0]).round(1).tolist()  # classical
[0.0, 1.0, 0.0]
>>> pp = ProjectionPursuitPCA(n_components=1).fit(X)
>>> abs(pp.components_[0]).round(1).tolist()
[1.0, 0.1, 0.1]

```

## Discriminant analysis

`RobustLDA` and `RobustQDA` are scikit-learn classifiers. Outliers in the
training data, including mislabeled observations, do not move the class
centers:

```pycon
>>> from robustmodels.multivariate import RobustLDA
>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(size=(50, 2)), rng.normal(size=(50, 2)) + 3])
>>> y = np.repeat([0, 1], 50)
>>> X[:5] = [15, 15]  # five outliers labeled as class 0
>>> lda = RobustLDA(random_state=0).fit(X, y)
>>> lda.means_.round(1).tolist()
[[-0.1, 0.2], [3.1, 3.0]]
>>> lda.predict([[0, 0], [3, 3]]).tolist()
[0, 1]
>>> X[y == 0].mean(axis=0).round(1).tolist()  # the classical center of class 0
[1.5, 1.7]

```

`predict_proba` gives posterior probabilities under the normal model and
`decision_function` the discriminant scores. `priors` sets the prior
probabilities; by default they are the class proportions, outliers included.
Use `RobustQDA` when the classes have different covariance matrices. For two
variables, [`plots.discriminant_plot`](plots.md#classification-regions) draws
the regions of either rule.

## Tests on mean vectors

`hotelling_t2` tests whether the center of a sample equals a given vector. For
paired samples, test the differences against zero.

```pycon
>>> from robustmodels.multivariate import hotelling_t2
>>> X = np.random.default_rng(0).normal(size=(60, 3))
>>> X[:6] += 8  # six outliers
>>> result = hotelling_t2(X, random_state=0)
>>> round(result.statistic, 2), round(result.pvalue, 2)
(5.16, 0.27)
>>> hotelling_t2(X, mu=[1, 1, 1], random_state=0).pvalue < 0.001
True

```

`manova` tests whether two or more groups share the same mean vector, assuming
a common covariance matrix. With two groups it is a robust two-sample
Hotelling test.

```pycon
>>> from robustmodels.multivariate import manova
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(90, 2))
>>> groups = np.repeat([0, 1, 2], 30)
>>> X[:4] += 9  # outliers in the first group
>>> result = manova(X, groups, random_state=0)
>>> round(result.statistic, 3), round(result.pvalue, 2)
(0.962, 0.63)
>>> X[groups == 2] += 1.5  # a real shift of the third group
>>> manova(X, groups, random_state=0).pvalue < 0.001
True

```

Both tests return named tuples. They assume that the regular observations are
multivariate normal: the p-values are approximations calibrated at that model,
not distribution-free results.

## Numerical limits and sources

**ROBPCA.** The outlyingness uses directions through pairs of observations: all
pairs when there are at most `n_directions`, otherwise a random selection, so
set `random_state` for reproducible results. When every pair is used, the fit
agrees with `rrcov::PcaHubert` to rounding error on the reference data sets
(center, loadings up to sign, eigenvalues and both distances). The orthogonal
cutoff follows the paper, with the univariate MCD of the distances to the power
2/3; rrcov uses their median and MAD, which gives a different cutoff. With all
components retained there are no orthogonal distances and the cutoff is 0.

**Projection pursuit.** The grid search stops at a local maximum of a
criterion that is not smooth. In the tests it reaches the maximum of a dense
search in two dimensions, and reproduces classical PCA when the scale is the
standard deviation. A cluster of outliers close to the data inflates the
robust scale in intermediate directions and tilts the first component, the
known bias of projection-pursuit estimators: on the plane example above the
fitted plane is 28 degrees away from that of the regular observations.
`scale="qn"` is more efficient than the default MAD and much slower for
thousands of observations.

**Discriminant analysis.** The common covariance is the reweighted MCD of the
observations centered by class (pooled observations), as the default of
`rrcov::Linda`; means and covariance match it to rounding error when both
searches find the same subsets. A class whose MCD covariance is singular, for
example because more than half of the values of a variable are tied, raises
`ValueError`.

**Hotelling's T².** The scaled F distribution uses the constants that Willems
et al. fitted to simulations of the reweighted MCD with coverage 0.75; the
statistic, degrees of freedom and p-value agree with `rrcov::T2.test` to
rounding error. The approximation is slightly liberal: in simulations with
normal data (`n = 40, p = 1`; `n = 50, p = 3`; `n = 100, p = 5`) the test
rejected 5.5%, 6.0% and 6.7% of the time at the 5% level.

**MANOVA.** There is no closed form for the null distribution of the robust
Wilks' lambda: it is simulated for the group sizes and dimension at hand, so
the p-value has Monte Carlo error and depends on `random_state`. Each simulated
sample needs one MCD per group and one for the pooled data, which is why the
deterministic MCD is used; 500 simulations of three groups of 30 take a few
seconds. With FAST-MCD the statistic agrees with `rrcov::Wilks.test` to
rounding error; with the deterministic search another subset, and another
lambda, can be found. In simulations with normal data, also strongly
correlated, the test rejected 3.7% to 6.2% of the time at the 5% level.

Sources: [ROBPCA](references.md#hrvb05), projection pursuit
([definition](references.md#lc85), [grid algorithm](references.md#cfo07)),
[robust discriminant analysis](references.md#hvd04) with the
[pooled-observations estimator](references.md#hf00),
[robust Hotelling test](references.md#wpra02) and
[robust MANOVA](references.md#tf10).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.multivariate

.. autoclass:: ROBPCA
   :members: fit, transform, inverse_transform, distances
.. autoclass:: ProjectionPursuitPCA
   :members: fit, transform, inverse_transform, distances
.. autoclass:: RobustLDA
   :members: fit, predict, predict_proba, decision_function
.. autoclass:: RobustQDA
   :members: fit, predict, predict_proba, decision_function
.. autofunction:: hotelling_t2
.. autofunction:: manova
```
