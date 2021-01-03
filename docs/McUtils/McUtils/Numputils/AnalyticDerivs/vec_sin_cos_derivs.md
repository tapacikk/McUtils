# <a id="McUtils.McUtils.Numputils.AnalyticDerivs.vec_sin_cos_derivs">vec_sin_cos_derivs</a>

Derivative of `sin(a, b)` and `cos(a, b)` with respect to both vector components

```python
vec_sin_cos_derivs(a, b, order=1, check_derivatives=False, zero_thresh=None): 
```

- `a`: `np.ndarray`
    >vector
- `order`: `int`
    >number of derivatives to return
- `zero_thresh`: `None | float`
    >threshold for when a norm should be called 0. for numerical reasons
- `:returns`: `list`
    >derivative tensors



