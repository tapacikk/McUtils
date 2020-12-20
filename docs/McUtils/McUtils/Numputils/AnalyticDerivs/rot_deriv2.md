# <a id="McUtils.McUtils.Numputils.AnalyticDerivs.rot_deriv2">rot_deriv2</a>

Gives a rotation matrix second derivative w/r/t some unspecified coordinate
    (you have to supply the chain rule terms)

```python
rot_deriv2(angle, axis, dAngle1, dAxis1, dAngle2, dAxis2, d2Angle, d2Axis): 
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

###Examples:
