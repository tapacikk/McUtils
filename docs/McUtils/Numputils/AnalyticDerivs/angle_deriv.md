# <a id="McUtils.Numputils.AnalyticDerivs.angle_deriv">angle_deriv</a>
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/AnalyticDerivs.py#L767)]
</div>

```python
angle_deriv(coords, i, j, k, order=1, zero_thresh=None): 
```
Gives the derivative of the angle between i, j, and k with respect to the Cartesians
- `coords`: `np.ndarray`
    >No description...
- `i`: `int | Iterable[int]`
    >index of the central atom
- `j`: `int | Iterable[int]`
    >index of one of the outside atoms
- `k`: `int | Iterable[int]`
    >index of the other outside atom
- `:returns`: `np.ndarray`
    >derivatives of the angle with respect to atoms i, j, and k 



___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/AnalyticDerivs.py#L767?message=Update%20Docs)