## <a id="McUtils.Data.CommonData.DataRecord">DataRecord</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L171)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L171?message=Update%20Docs)]
</div>

Represents an individual record that might be accessed from a `DataHandler`.
Implements _most_ of the `dict` interface, but, to make things a bit easier when
pickling, is not implemented as a proper subclass of `dict`.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Data.CommonData.DataRecord.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, data_handler, key, records): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L177)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L177?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.keys" class="docs-object-method">&nbsp;</a> 
```python
keys(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L182)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L182?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.values" class="docs-object-method">&nbsp;</a> 
```python
values(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L184)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L184?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.items" class="docs-object-method">&nbsp;</a> 
```python
items(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L186)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L186?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L189)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L189?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L192)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L192?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.__getstate__" class="docs-object-method">&nbsp;</a> 
```python
__getstate__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L200)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L200?message=Update%20Docs)]
</div>

<a id="McUtils.Data.CommonData.DataRecord.__setstate__" class="docs-object-method">&nbsp;</a> 
```python
__setstate__(self, state): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/CommonData.py#L205)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L205?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Data/CommonData/DataRecord.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Data/CommonData/DataRecord.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Data/CommonData/DataRecord.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Data/CommonData/DataRecord.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Data/CommonData.py#L171?message=Update%20Docs)