# <a id="McUtils.Numputils.AnalyticDerivs.rot_deriv2">rot_deriv2</a>
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/edit/Numputils/AnalyticDerivs.py#L131)]
</div>

```python
rot_deriv2(angle, axis, dAngle1, dAxis1, dAngle2, dAxis2, d2Angle, d2Axis): 
```
Gives a rotation matrix second derivative w/r/t some unspecified coordinate
    (you have to supply the chain rule terms)
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



___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Numputils/AnalyticDerivs/rot_deriv2.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/rot_deriv2.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Numputils/AnalyticDerivs/rot_deriv2.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/rot_deriv2.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/AnalyticDerivs.py#L131?message=Update%20Docs)