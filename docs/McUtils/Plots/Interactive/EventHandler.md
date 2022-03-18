## <a id="McUtils.Plots.Interactive.EventHandler">EventHandler</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L11)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L11?message=Update%20Docs)]
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
Event: type
```
<a id="McUtils.Plots.Interactive.EventHandler.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, figure, on_click=None, on_release=None, on_draw=None, on_key_press=None, on_key_release=None, on_move=None, on_select=None, on_resize=None, on_scroll=None, on_figure_entered=None, on_figure_left=None, on_axes_entered=None, on_axes_left=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L12)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L12?message=Update%20Docs)]
</div>

Creates an EventHandler on a Figure that handles most interactivity stuff
- `figure`: `GraphicsBase`
    >No description...
- `on_click`: `Any`
    >No description...
- `on_release`: `Any`
    >No description...
- `on_draw`: `Any`
    >No description...
- `on_key_press`: `Any`
    >No description...
- `on_key_release`: `Any`
    >No description...
- `on_move`: `Any`
    >No description...
- `on_select`: `Any`
    >No description...
- `on_resize`: `Any`
    >No description...
- `on_scroll`: `Any`
    >No description...
- `on_figure_entered`: `Any`
    >No description...
- `on_figure_left`: `Any`
    >No description...
- `on_axes_entered`: `Any`
    >No description...
- `on_axes_left`: `Any`
    >No description...

<a id="McUtils.Plots.Interactive.EventHandler.bind" class="docs-object-method">&nbsp;</a> 
```python
bind(self, **handlers): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L104)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L104?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.handlers" class="docs-object-method">&nbsp;</a> 
```python
@property
handlers(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.ButtonPressedEvent" class="docs-object-method">&nbsp;</a> 
```python
ButtonPressedEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L157)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L157?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.ButtonReleasedEvent" class="docs-object-method">&nbsp;</a> 
```python
ButtonReleasedEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L165)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L165?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.DrawEvent" class="docs-object-method">&nbsp;</a> 
```python
DrawEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L173)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L173?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.KeyPressedEvent" class="docs-object-method">&nbsp;</a> 
```python
KeyPressedEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L181)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L181?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.KeyReleasedEvent" class="docs-object-method">&nbsp;</a> 
```python
KeyReleasedEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L194)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L194?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.MoveEvent" class="docs-object-method">&nbsp;</a> 
```python
MoveEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L207)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L207?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.SelectEvent" class="docs-object-method">&nbsp;</a> 
```python
SelectEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L215)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L215?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.ScrollEvent" class="docs-object-method">&nbsp;</a> 
```python
ScrollEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L223)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L223?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.FigureEnterEvent" class="docs-object-method">&nbsp;</a> 
```python
FigureEnterEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L231)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L231?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.FigureLeaveEvent" class="docs-object-method">&nbsp;</a> 
```python
FigureLeaveEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L239)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L239?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.AxesEnterEvent" class="docs-object-method">&nbsp;</a> 
```python
AxesEnterEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L247)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L247?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Interactive.EventHandler.AxesLeaveEvent" class="docs-object-method">&nbsp;</a> 
```python
AxesLeaveEvent(self, handler, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Interactive.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L255?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Interactive/EventHandler.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Interactive/EventHandler.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Interactive/EventHandler.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Interactive/EventHandler.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Interactive.py#L11?message=Update%20Docs)