## <a id="McUtils.Plots.Graphics.GraphicsBase">GraphicsBase</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L27)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L27?message=Update%20Docs)]
</div>

The base class for all things Graphics
Defines the common parts of the interface with some calling into matplotlib

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
opt_keys: set
layout_keys: set
MPLManager: type
known_keys: set
```
<a id="McUtils.Plots.Graphics.GraphicsBase.resolve_figure_graphics" class="docs-object-method">&nbsp;</a> 
```python
resolve_figure_graphics(fig): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L34)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L34?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.add_figure_graphics" class="docs-object-method">&nbsp;</a> 
```python
add_figure_graphics(fig, graphics): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L38)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L38?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.remove_figure_mapping" class="docs-object-method">&nbsp;</a> 
```python
remove_figure_mapping(fig): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L47)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L47?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.get_child_graphics" class="docs-object-method">&nbsp;</a> 
```python
get_child_graphics(fig): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L53)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L53?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *args, figure=None, tighten=False, axes=None, subplot_kw=None, parent=None, image_size=None, padding=None, aspect_ratio=None, interactive=None, mpl_backend=None, theme=None, prop_manager=<class 'McUtils.Plots.Properties.GraphicsPropertyManager'>, theme_manager=<class 'McUtils.Plots.Styling.ThemeManager'>, managed=None, strict=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L130)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L130?message=Update%20Docs)]
</div>


- `args`: `Any`
    >No description...
- `figure`: `matplotlib.figure.Figure | None`
    >No description...
- `axes`: `matplotlib.axes.Axes | None`
    >No description...
- `subplot_kw`: `dict | None`
    >No description...
- `parent`: `GraphicsBase | None`
    >No description...
- `opts`: `Any`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.pyplot" class="docs-object-method">&nbsp;</a> 
```python
@property
pyplot(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.parent" class="docs-object-method">&nbsp;</a> 
```python
@property
parent(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.children" class="docs-object-method">&nbsp;</a> 
```python
@property
children(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.event_handlers" class="docs-object-method">&nbsp;</a> 
```python
@property
event_handlers(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.animated" class="docs-object-method">&nbsp;</a> 
```python
@property
animated(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.bind_events" class="docs-object-method">&nbsp;</a> 
```python
bind_events(self, *handlers, **events): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L521)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L521?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.create_animation" class="docs-object-method">&nbsp;</a> 
```python
create_animation(self, *args, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L535)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L535?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.set_options" class="docs-object-method">&nbsp;</a> 
```python
set_options(self, event_handlers=None, animated=None, prolog=None, epilog=None, strict=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L550)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L550?message=Update%20Docs)]
</div>

Sets options for the plot
- `event_handlers`: `Any`
    >No description...
- `animated`: `Any`
    >No description...
- `opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.prolog" class="docs-object-method">&nbsp;</a> 
```python
@property
prolog(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.epilog" class="docs-object-method">&nbsp;</a> 
```python
@property
epilog(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.copy_axes" class="docs-object-method">&nbsp;</a> 
```python
copy_axes(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L625)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L625?message=Update%20Docs)]
</div>

Copies the axes object
- `:returns`: `matplotlib.axes.Axes`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.refresh" class="docs-object-method">&nbsp;</a> 
```python
refresh(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L643)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L643?message=Update%20Docs)]
</div>

Refreshes the axes
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.opts" class="docs-object-method">&nbsp;</a> 
```python
@property
opts(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.copy" class="docs-object-method">&nbsp;</a> 
```python
copy(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L671)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L671?message=Update%20Docs)]
</div>

Creates a copy of the object with new axes and a new figure
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.change_figure" class="docs-object-method">&nbsp;</a> 
```python
change_figure(self, new, *init_args, figs=None, **init_kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L686)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L686?message=Update%20Docs)]
</div>

Creates a copy of the object with new axes and a new figure
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.prep_show" class="docs-object-method">&nbsp;</a> 
```python
prep_show(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L713)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L713?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.show" class="docs-object-method">&nbsp;</a> 
```python
show(self, reshow=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L722)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L722?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.close" class="docs-object-method">&nbsp;</a> 
```python
close(self, force=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L747)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L747?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.__del__" class="docs-object-method">&nbsp;</a> 
```python
__del__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L754)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L754?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.clear" class="docs-object-method">&nbsp;</a> 
```python
clear(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L760)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L760?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Graphics.GraphicsBase.savefig" class="docs-object-method">&nbsp;</a> 
```python
savefig(self, where, format=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L772)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L772?message=Update%20Docs)]
</div>

Saves the image to file
- `where`: `Any`
    >No description...
- `format`: `Any`
    >No description...
- `kw`: `Any`
    >No description...
- `:returns`: `str`
    >file it was saved to (I think...?)

<a id="McUtils.Plots.Graphics.GraphicsBase.to_png" class="docs-object-method">&nbsp;</a> 
```python
to_png(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L792)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L792?message=Update%20Docs)]
</div>

Used by Jupyter and friends to make a version of the image that they can display, hence the extra 'tight_layout' call
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Graphics.GraphicsBase.add_colorbar" class="docs-object-method">&nbsp;</a> 
```python
add_colorbar(self, graphics=None, norm=None, cmap=None, size=(20, 200), tick_padding=40, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L814)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L814?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Graphics/GraphicsBase.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Graphics/GraphicsBase.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Graphics/GraphicsBase.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Graphics/GraphicsBase.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L27?message=Update%20Docs)