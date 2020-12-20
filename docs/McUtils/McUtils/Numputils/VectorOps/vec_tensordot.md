# <a id="McUtils.McUtils.Numputils.VectorOps.vec_tensordot">vec_tensordot</a>

Defines a version of tensordot that uses matmul to operate over stacks of things
    Basically had to duplicate the code for regular tensordot but then change the final call

```python
vec_tensordot(tensa, tensb, axes=2, shared=None): 
```

- `tensa`: `Any`
    >No description...
- `tensb`: `Any`
    >No description...
- `axes`: `Any`
    >No description...
- `shared`: `int | None`
    >the axes that should be treated as shared (for now just an int)
- `:returns`: `_`
    >No description...

###Examples:
