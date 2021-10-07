## <a id="McUtils.Zachary.Mesh.Mesh">Mesh</a>
A general Mesh class representing data points in n-dimensions
in either a structured, unstructured, or semi-structured manner.
Exists mostly to provides a unified interface to difference FD and Surface methods.

### Properties and Methods
```python
MeshError: type
MeshType: EnumMeta
```
<a id="McUtils.Zachary.Mesh.Mesh.__new__" class="docs-object-method">&nbsp;</a>
```python
__new__(cls, data, mesh_type=None, allow_indeterminate=None): 
```

- `griddata`: `np.ndarray`
    >the raw grid-point data the mesh uses
- `mesh_type`: `None | str`
    >the type of mesh we have
- `opts`: `Any`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, *args, **kwargs): 
```

<a id="McUtils.Zachary.Mesh.Mesh.__array_finalize__" class="docs-object-method">&nbsp;</a>
```python
__array_finalize__(self, mesh): 
```

<a id="McUtils.Zachary.Mesh.Mesh.mesh_spacings" class="docs-object-method">&nbsp;</a>
```python
@property
mesh_spacings(self): 
```

<a id="McUtils.Zachary.Mesh.Mesh.subgrids" class="docs-object-method">&nbsp;</a>
```python
@property
subgrids(self): 
```

<a id="McUtils.Zachary.Mesh.Mesh.dimension" class="docs-object-method">&nbsp;</a>
```python
@property
dimension(self): 
```
Returns the dimension of the grid (not necessarily ndim)
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.npoints" class="docs-object-method">&nbsp;</a>
```python
@property
npoints(self): 
```
Returns the number of gridpoints in the mesh
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.gridpoints" class="docs-object-method">&nbsp;</a>
```python
@property
gridpoints(self): 
```
Returns the flattened set of gridpoints for a structured tensor grid and otherwise just returns the gridpoints
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_npoints" class="docs-object-method">&nbsp;</a>
```python
get_npoints(g): 
```
Returns the number of gridpoints in the grid
- `g`: `np.ndarray`
    >No description...
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_gridpoints" class="docs-object-method">&nbsp;</a>
```python
get_gridpoints(g): 
```
Returns the gridpoints in the grid
- `g`: `np.ndarray`
    >No description...
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_mesh_subgrids" class="docs-object-method">&nbsp;</a>
```python
get_mesh_subgrids(grid, tol=8): 
```
Returns the subgrids for a mesh
- `grid`: `Any`
    >No description...
- `tol`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_mesh_spacings" class="docs-object-method">&nbsp;</a>
```python
get_mesh_spacings(grid, tol=8): 
```

<a id="McUtils.Zachary.Mesh.Mesh.get_mesh_type" class="docs-object-method">&nbsp;</a>
```python
get_mesh_type(grid, check_product_grid=True, check_regular_grid=True, tol=8): 
```
Determines what kind of grid we're working with
- `grid`: `np.ndarray`
    >No description...
- `:returns`: `MeshType`
    >mesh_type

<a id="McUtils.Zachary.Mesh.Mesh.RegularMesh" class="docs-object-method">&nbsp;</a>
```python
RegularMesh(*mesh_specs): 
```
Builds a grid from multiple linspace arguments,
        basically insuring it's structured (if non-Empty)
- `mesh_specs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/Mesh/Mesh.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/Mesh/Mesh.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/Mesh/Mesh.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/Mesh/Mesh.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Mesh.py?message=Update%20Docs)