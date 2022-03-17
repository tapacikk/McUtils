## <a id="McUtils.Zachary.LazyTensors.TensorOp">TensorOp</a>
A lazy representation of tensor operations to save memory

### Properties and Methods
<a id="McUtils.Zachary.LazyTensors.TensorOp.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, a, b, axis=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/LazyTensors.py#L130)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/LazyTensors.py#L130?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.LazyTensors.TensorOp.op" class="docs-object-method">&nbsp;</a> 
```python
op(self, a, b): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/LazyTensors.py#L148)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/LazyTensors.py#L148?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.LazyTensors.TensorOp.get_shape" class="docs-object-method">&nbsp;</a> 
```python
get_shape(self, a, b): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/LazyTensors.py#L152)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/LazyTensors.py#L152?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.LazyTensors.TensorOp.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/LazyTensors.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/LazyTensors.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.LazyTensors.TensorOp.array" class="docs-object-method">&nbsp;</a> 
```python
@property
array(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/LazyTensors.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/LazyTensors.py#L?message=Update%20Docs)]
</div>

Ought to always compile down to a proper ndarray
- `:returns`: `np.ndarray`
    >No description...

<a id="McUtils.Zachary.LazyTensors.TensorOp.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, i): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/LazyTensors.py#L165)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/LazyTensors.py#L165?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/LazyTensors/TensorOp.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/LazyTensors/TensorOp.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/LazyTensors/TensorOp.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/LazyTensors/TensorOp.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/LazyTensors.py?message=Update%20Docs)