## <a id="McUtils.Plots.Graphics.GraphicsBase">GraphicsBase</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L29)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L29?message=Update%20Docs)]
</div>

The base class for all things Graphics
Defines the common parts of the interface with some calling into matplotlib







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 ```python
opt_keys: set
layout_keys: set
MPLManager: MPLManager
known_keys: set
axes_params: set
inset_options: dict
axes_keys: set
```
<a id="McUtils.Plots.Graphics.GraphicsBase.resolve_figure_graphics" class="docs-object-method">&nbsp;</a> 
```python
resolve_figure_graphics(fig): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L36)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L36?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.add_figure_graphics" class="docs-object-method">&nbsp;</a> 
```python
add_figure_graphics(fig, graphics): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L40)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L40?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.remove_figure_mapping" class="docs-object-method">&nbsp;</a> 
```python
remove_figure_mapping(fig): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L49)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L49?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.get_child_graphics" class="docs-object-method">&nbsp;</a> 
```python
get_child_graphics(fig): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L55)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L55?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.resolve_axes_graphics" class="docs-object-method">&nbsp;</a> 
```python
resolve_axes_graphics(axes): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L63)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L63?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.add_axes_graphics" class="docs-object-method">&nbsp;</a> 
```python
add_axes_graphics(axes, graphics): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L68)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L68?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.remove_axes_mapping" class="docs-object-method">&nbsp;</a> 
```python
remove_axes_mapping(axes): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L77)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L77?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.get_axes_child_graphics" class="docs-object-method">&nbsp;</a> 
```python
get_axes_child_graphics(axes): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L84)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L84?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.get_raw_attr" class="docs-object-method">&nbsp;</a> 
```python
get_raw_attr(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L122)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L122?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *args, name=None, figure=None, tighten=False, axes=None, subplot_kw=None, parent=None, image_size=None, padding=None, aspect_ratio=None, interactive=None, reshowable=None, mpl_backend=None, theme=None, prop_manager=<class 'McUtils.Plots.Properties.GraphicsPropertyManager'>, theme_manager=<class 'McUtils.Plots.Styling.ThemeManager'>, managed=None, strict=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L203)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L203?message=Update%20Docs)]
</div>

  - `args`: `Any`
    > 
  - `figure`: `matplotlib.figure.Figure | None`
    > 
  - `axes`: `matplotlib.axes.Axes | None`
    > 
  - `subplot_kw`: `dict | None`
    > 
  - `parent`: `GraphicsBase | None`
    > 
  - `opts`: `Any`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.pyplot" class="docs-object-method">&nbsp;</a> 
```python
@property
pyplot(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L540)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L540?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.parent" class="docs-object-method">&nbsp;</a> 
```python
@property
parent(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L581)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L581?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.figure_parent" class="docs-object-method">&nbsp;</a> 
```python
@property
figure_parent(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L587)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L587?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.inset" class="docs-object-method">&nbsp;</a> 
```python
@property
inset(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L590)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L590?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.children" class="docs-object-method">&nbsp;</a> 
```python
@property
children(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L594)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L594?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.event_handlers" class="docs-object-method">&nbsp;</a> 
```python
@property
event_handlers(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L604)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L604?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.animated" class="docs-object-method">&nbsp;</a> 
```python
@property
animated(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L612)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L612?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.bind_events" class="docs-object-method">&nbsp;</a> 
```python
bind_events(self, *handlers, **events): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L616)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L616?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.create_animation" class="docs-object-method">&nbsp;</a> 
```python
create_animation(self, *args, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L630)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L630?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.set_options" class="docs-object-method">&nbsp;</a> 
```python
set_options(self, event_handlers=None, animated=None, prolog=None, epilog=None, strict=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L645)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L645?message=Update%20Docs)]
</div>
Sets options for the plot
  - `event_handlers`: `Any`
    > 
  - `animated`: `Any`
    > 
  - `opts`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.prolog" class="docs-object-method">&nbsp;</a> 
```python
@property
prolog(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L678)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L678?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.epilog" class="docs-object-method">&nbsp;</a> 
```python
@property
epilog(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L687)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L687?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.copy_axes" class="docs-object-method">&nbsp;</a> 
```python
copy_axes(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L720)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L720?message=Update%20Docs)]
</div>
Copies the axes object
  - `:returns`: `matplotlib.axes.Axes`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.refresh" class="docs-object-method">&nbsp;</a> 
```python
refresh(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L738)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L738?message=Update%20Docs)]
</div>
Refreshes the axes
  - `:returns`: `_`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.opts" class="docs-object-method">&nbsp;</a> 
```python
@property
opts(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L751)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L751?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.copy" class="docs-object-method">&nbsp;</a> 
```python
copy(self, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L767)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L767?message=Update%20Docs)]
</div>
Creates a copy of the object with new axes and a new figure
  - `:returns`: `_`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.change_figure" class="docs-object-method">&nbsp;</a> 
```python
change_figure(self, new, *init_args, figs=None, **init_kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L782)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L782?message=Update%20Docs)]
</div>
Creates a copy of the object with new axes and a new figure
  - `:returns`: `_`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.prep_show" class="docs-object-method">&nbsp;</a> 
```python
prep_show(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L815)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L815?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.show" class="docs-object-method">&nbsp;</a> 
```python
show(self, reshow=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L825)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L825?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.close" class="docs-object-method">&nbsp;</a> 
```python
close(self, force=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L854)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L854?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.__del__" class="docs-object-method">&nbsp;</a> 
```python
__del__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L873)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L873?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L879)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L879?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.clear" class="docs-object-method">&nbsp;</a> 
```python
clear(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L887)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L887?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.savefig" class="docs-object-method">&nbsp;</a> 
```python
savefig(self, where, format=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L906)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L906?message=Update%20Docs)]
</div>
Saves the image to file
  - `where`: `Any`
    > 
  - `format`: `Any`
    > 
  - `kw`: `Any`
    > 
  - `:returns`: `str`
    > f
i
l
e
 
i
t
 
w
a
s
 
s
a
v
e
d
 
t
o
 
(
I
 
t
h
i
n
k
.
.
.
?
)


<a id="McUtils.Plots.Graphics.GraphicsBase.to_png" class="docs-object-method">&nbsp;</a> 
```python
to_png(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L927)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L927?message=Update%20Docs)]
</div>
Used by Jupyter and friends to make a version of the image that they can display, hence the extra 'tight_layout' call
  - `:returns`: `_`
    >


<a id="McUtils.Plots.Graphics.GraphicsBase.add_colorbar" class="docs-object-method">&nbsp;</a> 
```python
add_colorbar(self, graphics=None, norm=None, cmap=None, size=(20, 200), tick_padding=40, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L949)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L949?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsBase.create_inset" class="docs-object-method">&nbsp;</a> 
```python
create_inset(self, bbox, coordinates='scaled', graphics_class=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsBase.py#L1019)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsBase.py#L1019?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Plots/Graphics/GraphicsBase.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Plots/Graphics/GraphicsBase.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Plots/Graphics/GraphicsBase.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Plots/Graphics/GraphicsBase.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L29?message=Update%20Docs)   
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