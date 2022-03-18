## <a id="McUtils.Zachary.Mesh.Mesh">Mesh</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L22)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L22?message=Update%20Docs)]
</div>

A general Mesh class representing data points in n-dimensions
in either a structured, unstructured, or semi-structured manner.
Exists mostly to provides a unified interface to difference FD and Surface methods.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
MeshError: type
MeshType: EnumMeta
```
<a id="McUtils.Zachary.Mesh.Mesh.__new__" class="docs-object-method">&nbsp;</a> 
```python
__new__(cls, data, mesh_type=None, allow_indeterminate=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L35)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L35?message=Update%20Docs)]
</div>


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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L63)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L63?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Mesh.Mesh.__array_finalize__" class="docs-object-method">&nbsp;</a> 
```python
__array_finalize__(self, mesh): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L67)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L67?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Mesh.Mesh.mesh_spacings" class="docs-object-method">&nbsp;</a> 
```python
@property
mesh_spacings(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Mesh.Mesh.subgrids" class="docs-object-method">&nbsp;</a> 
```python
@property
subgrids(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Mesh.Mesh.dimension" class="docs-object-method">&nbsp;</a> 
```python
@property
dimension(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L?message=Update%20Docs)]
</div>

Returns the dimension of the grid (not necessarily ndim)
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.npoints" class="docs-object-method">&nbsp;</a> 
```python
@property
npoints(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L?message=Update%20Docs)]
</div>

Returns the number of gridpoints in the mesh
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.gridpoints" class="docs-object-method">&nbsp;</a> 
```python
@property
gridpoints(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L?message=Update%20Docs)]
</div>

Returns the flattened set of gridpoints for a structured tensor grid and otherwise just returns the gridpoints
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_npoints" class="docs-object-method">&nbsp;</a> 
```python
get_npoints(g): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L124)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L124?message=Update%20Docs)]
</div>

Returns the number of gridpoints in the grid
- `g`: `np.ndarray`
    >No description...
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_gridpoints" class="docs-object-method">&nbsp;</a> 
```python
get_gridpoints(g): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L134)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L134?message=Update%20Docs)]
</div>

Returns the gridpoints in the grid
- `g`: `np.ndarray`
    >No description...
- `:returns`: `int`
    >No description...

<a id="McUtils.Zachary.Mesh.Mesh.get_mesh_subgrids" class="docs-object-method">&nbsp;</a> 
```python
get_mesh_subgrids(grid, tol=8): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L145)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L145?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L171)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L171?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Mesh.Mesh.get_mesh_type" class="docs-object-method">&nbsp;</a> 
```python
get_mesh_type(grid, check_product_grid=True, check_regular_grid=True, tol=8): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L200)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L200?message=Update%20Docs)]
</div>

Determines what kind of grid we're working with
- `grid`: `np.ndarray`
    >No description...
- `:returns`: `MeshType`
    >mesh_type

<a id="McUtils.Zachary.Mesh.Mesh.RegularMesh" class="docs-object-method">&nbsp;</a> 
```python
RegularMesh(*mesh_specs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Mesh.py#L292)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L292?message=Update%20Docs)]
</div>

Builds a grid from multiple linspace arguments,
        basically insuring it's structured (if non-Empty)
- `mesh_specs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Mesh/Mesh.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Mesh/Mesh.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Mesh/Mesh.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Mesh/Mesh.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Mesh.py#L22?message=Update%20Docs)