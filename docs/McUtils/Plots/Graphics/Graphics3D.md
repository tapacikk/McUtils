## <a id="McUtils.Plots.Graphics.Graphics3D">Graphics3D</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L1153)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1153?message=Update%20Docs)]
</div>

Extends the standard matplotlib 3D plotting to use all the Graphics extensions

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
opt_keys: set
```
<a id="McUtils.Plots.Graphics.Graphics3D.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *args, figure=None, axes=None, subplot_kw=None, event_handlers=None, animate=None, axes_labels=None, plot_label=None, plot_range=None, plot_legend=None, ticks=None, scale=None, ticks_style=None, image_size=None, background=None, view_settings=None, backend=<Backends.MPL: 'matplotlib'>, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L1158)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1158?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.set_options" class="docs-object-method">&nbsp;</a> 
```python
set_options(self, view_settings=None, **parent_opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L1199)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1199?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.load_mpl" class="docs-object-method">&nbsp;</a> 
```python
load_mpl(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L1214)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1214?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.plot_label" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_label(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.plot_legend" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_legend(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.axes_labels" class="docs-object-method">&nbsp;</a> 
```python
@property
axes_labels(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.frame" class="docs-object-method">&nbsp;</a> 
```python
@property
frame(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.plot_range" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_range(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.ticks" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.ticks_style" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.ticks_label_style" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks_label_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.scale" class="docs-object-method">&nbsp;</a> 
```python
@property
scale(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.aspect_ratio" class="docs-object-method">&nbsp;</a> 
```python
@property
aspect_ratio(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.image_size" class="docs-object-method">&nbsp;</a> 
```python
@property
image_size(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.padding" class="docs-object-method">&nbsp;</a> 
```python
@property
padding(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.background" class="docs-object-method">&nbsp;</a> 
```python
@property
background(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.colorbar" class="docs-object-method">&nbsp;</a> 
```python
@property
colorbar(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.box_ratios" class="docs-object-method">&nbsp;</a> 
```python
@property
box_ratios(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.Graphics3D.view_settings" class="docs-object-method">&nbsp;</a> 
```python
@property
view_settings(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Graphics/Graphics3D.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Graphics/Graphics3D.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Graphics/Graphics3D.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Graphics/Graphics3D.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1153?message=Update%20Docs)