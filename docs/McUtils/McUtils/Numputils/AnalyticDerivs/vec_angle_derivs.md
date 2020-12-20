# <a id="McUtils.McUtils.Numputils.AnalyticDerivs.vec_angle_derivs">vec_angle_derivs</a>

Returns the derivatives of the angle between `a` and `b` with respect to their components

```python
vec_angle_derivs(a, b, order=1, up_vectors=None, zero_thresh=None): 
```

- `a`: `np.ndarray`
    >vector
- `b`: `np.ndarray`
    >vector
- `order`: `int`
    >order of derivatives to go up to
- `zero_thresh`: `float | None`
    >threshold for what is zero in a vector norm
- `:returns`: `list`
    >derivative tensors

###Examples:
