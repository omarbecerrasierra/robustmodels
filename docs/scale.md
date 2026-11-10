# Scale and skewness

Robust alternatives to the standard deviation, and robust measures of skewness
and tail weight, in `robustmodels.scale`.

## Choosing an estimator

| Function | Breakdown point | Gaussian efficiency | Notes |
| --- | --- | --- | --- |
| `mad` | 50% | 37% | Simplest; needs a center |
| `qn` | 50% | 82% | No center needed; a good default |
| `sn` | 50% | 58% | No center needed |
| `pn` | 13.4% | 86% | Most efficient, least robust |
| `tau_scale` | | | The scale used by `OGK` |
| `m_scale` | `breakdown` | | The scale minimized by S-estimators |

The consistency calibration targets the normal standard deviation
asymptotically. Efficiency and breakdown percentages in this table are
asymptotic values, not finite-sample guarantees. `m_scale` is always calibrated;
the other scale estimators expose `consistency=True`.

## Use

```pycon
>>> from robustmodels import scale
>>> x = [1, 2, 2, 3, 4, 100]
>>> round(scale.mad(x), 4)
1.4826
>>> round(scale.qn(x), 4)
1.3586
>>> round(scale.sn(x), 4)
1.1843
>>> round(scale.tau_scale(x), 4)
1.5998
>>> round(scale.m_scale(x), 4)
1.5019

```

One outlier in six is 16.7% of the sample, above the breakdown point of Pn:

```pycon
>>> round(scale.pn(x), 4)
54.6572

```

### Corrections

`consistency=False` disables the normal-consistency multiplier. For Qn, Sn and
Pn the finite correction is independent: disable `finite_correction` as well
to obtain the raw order statistic. For example, the default uncorrected MAD is:

```pycon
>>> scale.mad(x, consistency=False)
1.0

```

Qn, Sn and Pn also apply a finite-sample correction by default, which reduces
their bias in small normal samples; `mad` offers it with
`finite_correction=True`. Disable it to get the asymptotic version:

```pycon
>>> round(scale.qn(x, finite_correction=False), 4)
2.2191

```

### Skewness and tail weight

The medcouple lies in [-1, 1] and is positive for right-skewed data. The left
and right medcouples measure the weight of each tail.

```pycon
>>> y = [0.2, 0.5, 0.9, 1.1, 1.6, 2.4, 3.8, 7.5]
>>> round(scale.medcouple(y), 4)
0.4424
>>> round(scale.lmc(y), 4), round(scale.rmc(y), 4)
(0.0556, 0.2458)

```

## Numerical conventions and sources

The small-sample factors are fitted or simulated approximations; they do not
make every finite sample unbiased. They target normal data and the documented
center/quantile conventions. MAD's finite factors assume the sample median as
center; they are not calibrated for an arbitrary supplied center.

Qn uses the exact normal-quantile constant instead of robustbase's rounded
2.21914. robustbase's Qn C routine also casts some selection pivots to single
precision, so even unscaled results can differ around `1e-8`. Sn uses the
low median of high medians, not averages of middle distances.

Pn uses inverse empirical quantiles (`type=1` in R), including the author's
finite correction factors. Its normal constant is `1/(sqrt(2)*Phi^-1(3/4))`;
the author's reference R code uses the rounded reciprocal `1/0.9539`.

For the medcouple, the tie kernel follows Brys et al. (2004), equations
(2.1)–(2.3). An even number of kernels uses the average of the two middle
values. For `[-3, 0, 0]`, this definition gives `-0.5`, while robustbase 0.99-7
returns `-1`; the same issue can propagate to `lmc` and `rmc`.

The current generic sorted-row selector used by Qn, Pn and medcouple has an
`O(n log² n)` time bound and `O(n)` auxiliary storage. Sn uses `O(n log n)` time.
The existence of an `O(n log n)` algorithm in a paper does not establish that
bound for a different implementation. "Exact" order-statistic calculations
still use floating-point arithmetic.

Sources: [Rousseeuw–Croux](references.md#rc93),
[Tarr–Müller–Weber](references.md#tmw12),
[Brys–Hubert–Struyf](references.md#bhs04),
[tail weights](references.md#bhs06), and [Maronna–Zamar](references.md#mz02).
The [source map](references.md#method-to-source-map) covers every public method.

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.scale

.. autofunction:: mad
.. autofunction:: qn
.. autofunction:: sn
.. autofunction:: pn
.. autofunction:: tau_scale
.. autofunction:: m_scale
.. autofunction:: medcouple
.. autofunction:: lmc
.. autofunction:: rmc
```
