# <a id="McUtils.McUtils.Numputils.AnalyticDerivs.dihed_deriv">dihed_deriv</a>

Gives the derivative of the dihedral between i, j, k, and l with respect to the Cartesians
    Currently gives what are sometimes called the `psi` angles.
    Will be extended to also support more traditional `phi` angles

```python
dihed_deriv(coords, i, j, k, l, order=1, zero_thresh=None): 
```

- `coords`: `np.ndarray`
    >No description...
- `i`: `int | Iterable[int]`
    >No description...
- `j`: `int | Iterable[int]`
    >No description...
- `k`: `int | Iterable[int]`
    >No description...
- `l`: `int | Iterable[int]`
    >No description...
- `:returns`: `np.ndarray`
    >derivatives of the dihedral with respect to atoms i, j, k, and l

###Examples:
