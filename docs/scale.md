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

All of them estimate the standard deviation at the normal model
(`consistency=True`).

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

`consistency=False` returns the raw statistic, without the factor that makes it
estimate the standard deviation:

```pycon
>>> scale.mad(x, consistency=False)
1.0

```

Qn, Sn and Pn also apply a finite-sample correction by default, which removes
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
