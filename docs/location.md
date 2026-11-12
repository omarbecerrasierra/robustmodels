# Robust location estimators

Robust alternatives to the sample mean, in `robustmodels.location`.

## Choosing an estimator

| Function | Breakdown point | Gaussian efficiency | Notes |
| --- | --- | --- | --- |
| `hodges_lehmann` | 29.3% | 95.5% | No tuning; a good default |
| `m_location` (Huber) | 50% | 95% | Monotone psi; flat sets of solutions are possible |
| `m_location` (bisquare) | 50% | 95% | Rejects gross outliers completely |
| `tau_location` | 50% | | One weighted mean; no iteration |
| `trimmed_mean` | `proportion` | | Removes the tails |
| `winsorized_mean` | `proportion` | | Clips the tails |

The percentages are asymptotic under the relevant assumptions. The M-location
values assume the default robust scale and tuning. The median itself
(`numpy.median`) has a 50% asymptotic breakdown point and 64% Gaussian efficiency.

## Use

```pycon
>>> from robustmodels import location
>>> x = [1, 2, 2, 3, 4, 100]
>>> location.hodges_lehmann(x)
3.0
>>> round(location.m_location(x), 4)
2.7988
>>> round(location.m_location(x, psi="bisquare"), 4)
2.3869
>>> round(location.tau_location(x), 4)
2.3918

```

Trimming works with whole observations: it removes `floor(proportion * n)` values
from each end, so the default 10% removes nothing from a sample of six.

```pycon
>>> round(location.trimmed_mean(x), 4)
18.6667
>>> location.trimmed_mean(x, proportion=0.2)
2.75
>>> round(location.winsorized_mean(x, proportion=0.2), 4)
2.8333

```

`m_location` keeps the scale fixed. By default it is the MAD; pass another
estimate with `scale`:

```pycon
>>> from robustmodels import scale
>>> round(location.m_location(x, scale=scale.qn(x)), 4)
2.7655

```

## Numerical conventions and sources

`m_location` uses a fixed scale and starts from the median. A zero scale returns
the median. Bisquare can have multiple stationary solutions; a local iteration
does not certify a global minimum. If the requested iteration limit is reached,
the last location is returned with a `RuntimeWarning`. Huber's monotone score
can also have an interval of roots rather than a unique root.

The Hodges–Lehmann estimator includes diagonal pairs. Its current sorted-row
selector uses `O(n log² n)` time and `O(n)` auxiliary storage. It computes the
specified median in floating-point arithmetic.

Sources: [Hodges–Lehmann](references.md#hl63), [Huber](references.md#h64),
[Beaton–Tukey](references.md#bt74), [Tukey–McLaughlin](references.md#tm63),
and [Maronna–Zamar](references.md#mz02).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.location

.. autofunction:: hodges_lehmann
.. autofunction:: m_location
.. autofunction:: tau_location
.. autofunction:: trimmed_mean
.. autofunction:: winsorized_mean
```
