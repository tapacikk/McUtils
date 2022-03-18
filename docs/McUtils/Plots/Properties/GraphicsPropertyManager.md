## <a id="McUtils.Plots.Properties.GraphicsPropertyManager">GraphicsPropertyManager</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L12)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L12?message=Update%20Docs)]
</div>

Manages properties for Graphics objects so that concrete GraphicsBase instances don't need to duplicate code, but
at the same time things that build off of GraphicsBase don't need to implement all of these properties

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, graphics, figure, axes, managed=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L18)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L18?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.figure_label" class="docs-object-method">&nbsp;</a> 
```python
@property
figure_label(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.plot_label" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_label(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.plot_legend" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_legend(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.axes_labels" class="docs-object-method">&nbsp;</a> 
```python
@property
axes_labels(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.plot_range" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_range(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.ticks" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.ticks_style" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.frame_style" class="docs-object-method">&nbsp;</a> 
```python
@property
frame_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.ticks_label_style" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks_label_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.aspect_ratio" class="docs-object-method">&nbsp;</a> 
```python
@property
aspect_ratio(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.image_size" class="docs-object-method">&nbsp;</a> 
```python
@property
image_size(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.background" class="docs-object-method">&nbsp;</a> 
```python
@property
background(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.frame" class="docs-object-method">&nbsp;</a> 
```python
@property
frame(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.scale" class="docs-object-method">&nbsp;</a> 
```python
@property
scale(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.padding" class="docs-object-method">&nbsp;</a> 
```python
@property
padding(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.padding_left" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_left(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.padding_right" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_right(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.padding_top" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_top(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.padding_bottom" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_bottom(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.spacings" class="docs-object-method">&nbsp;</a> 
```python
@property
spacings(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Properties.GraphicsPropertyManager.colorbar" class="docs-object-method">&nbsp;</a> 
```python
@property
colorbar(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Properties.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Properties/GraphicsPropertyManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Properties/GraphicsPropertyManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Properties/GraphicsPropertyManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Properties/GraphicsPropertyManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Properties.py#L12?message=Update%20Docs)