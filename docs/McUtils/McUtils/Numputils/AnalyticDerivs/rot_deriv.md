# <a id="McUtils.McUtils.Numputils.AnalyticDerivs.rot_deriv">rot_deriv</a>

Gives a rotational derivative w/r/t some unspecified coordinate
    (you have to supply the chain rule terms)
    Assumes that axis is a unit vector.

```python
rot_deriv(angle, axis, dAngle, dAxis): 
```

- `angle`: `float`
    >angle for rotation
- `axis`: `np.ndarray`
    >axis for rotation
- `dAngle`: `float`
    >chain rule angle deriv.
- `dAxis`: `np.ndarray`
    >chain rule axis deriv.
- `:returns`: `np.ndarray`
    >derivatives of the rotation matrices with respect to both the angle and the axis



