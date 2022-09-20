## <a id="McUtils.Plots.Graphics.Graphics3D">Graphics3D</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L1347)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1347?message=Update%20Docs)]
</div>

Extends the standard matplotlib 3D plotting to use all the Graphics extensions







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 ```python
opt_keys: set
```
<a id="McUtils.Plots.Graphics.Graphics3D.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *args, figure=None, axes=None, subplot_kw=None, event_handlers=None, animate=None, axes_labels=None, plot_label=None, plot_range=None, plot_legend=None, ticks=None, scale=None, ticks_style=None, image_size=None, background=None, view_settings=None, backend=<Backends.MPL: 'matplotlib'>, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1352)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1352?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.set_options" class="docs-object-method">&nbsp;</a> 
```python
set_options(self, view_settings=None, **parent_opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1393)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1393?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.load_mpl" class="docs-object-method">&nbsp;</a> 
```python
load_mpl(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1408)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1408?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.plot_label" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_label(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1457)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1457?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.plot_legend" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_legend(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1464)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1464?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.axes_labels" class="docs-object-method">&nbsp;</a> 
```python
@property
axes_labels(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1471)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1471?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.frame" class="docs-object-method">&nbsp;</a> 
```python
@property
frame(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1478)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1478?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.plot_range" class="docs-object-method">&nbsp;</a> 
```python
@property
plot_range(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1485)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1485?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.ticks" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1556)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1556?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.ticks_style" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1499)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1499?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.ticks_label_style" class="docs-object-method">&nbsp;</a> 
```python
@property
ticks_label_style(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1506)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1506?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.scale" class="docs-object-method">&nbsp;</a> 
```python
@property
scale(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1513)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1513?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.aspect_ratio" class="docs-object-method">&nbsp;</a> 
```python
@property
aspect_ratio(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1577)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1577?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.image_size" class="docs-object-method">&nbsp;</a> 
```python
@property
image_size(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1527)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1527?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.padding" class="docs-object-method">&nbsp;</a> 
```python
@property
padding(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1534)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1534?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.background" class="docs-object-method">&nbsp;</a> 
```python
@property
background(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1541)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1541?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.colorbar" class="docs-object-method">&nbsp;</a> 
```python
@property
colorbar(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1548)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1548?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.box_ratios" class="docs-object-method">&nbsp;</a> 
```python
@property
box_ratios(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1563)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1563?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.Graphics3D.view_settings" class="docs-object-method">&nbsp;</a> 
```python
@property
view_settings(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/Graphics3D.py#L1570)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/Graphics3D.py#L1570?message=Update%20Docs)]
</div>
 </div>
</div>












---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Graphics/Graphics3D.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Graphics/Graphics3D.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Graphics/Graphics3D.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Graphics/Graphics3D.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1347?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>