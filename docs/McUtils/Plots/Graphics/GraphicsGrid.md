## <a id="McUtils.Plots.Graphics.GraphicsGrid">GraphicsGrid</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics.py#L1670)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1670?message=Update%20Docs)]
</div>

A class for easily building sophisticated multi-panel figures.
Robustification work still needs to be done, but the core interface is there.
Supports themes & direct, easy access to the panels, among other things.
Builds off of `GraphicsBase`.







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse " id="methods" markdown="1">
 ```python
default_style: dict
layout_keys: set
known_keys: set
GraphicsStack: GraphicsStack
```
<a id="McUtils.Plots.Graphics.GraphicsGrid.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *args, nrows=None, ncols=None, graphics_class=<class 'McUtils.Plots.Graphics.Graphics'>, figure=None, axes=None, subplot_kw=None, _subplot_init=None, mpl_backend=None, subimage_size=(310, 310), subimage_aspect_ratio='auto', padding=None, spacings=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1684)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1684?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.set_options" class="docs-object-method">&nbsp;</a> 
```python
set_options(self, padding=None, spacings=None, background=None, colorbar=None, figure_label=None, **parent_opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1884)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1884?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.__iter__" class="docs-object-method">&nbsp;</a> 
```python
__iter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1908)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1908?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1910)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1910?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.__setitem__" class="docs-object-method">&nbsp;</a> 
```python
__setitem__(self, item, val): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1917)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1917?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.set_image" class="docs-object-method">&nbsp;</a> 
```python
set_image(self, pos, val, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1930)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1930?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.calc_image_size" class="docs-object-method">&nbsp;</a> 
```python
calc_image_size(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1955)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1955?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.image_size" class="docs-object-method">&nbsp;</a> 
```python
@property
image_size(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1982)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1982?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.figure_label" class="docs-object-method">&nbsp;</a> 
```python
@property
figure_label(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1992)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1992?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.padding" class="docs-object-method">&nbsp;</a> 
```python
@property
padding(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L1998)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L1998?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.padding_left" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_left(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2004)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2004?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.padding_right" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_right(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2010)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2010?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.padding_top" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_top(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2016)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2016?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.padding_bottom" class="docs-object-method">&nbsp;</a> 
```python
@property
padding_bottom(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2022)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2022?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.spacings" class="docs-object-method">&nbsp;</a> 
```python
@property
spacings(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2029)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2029?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.background" class="docs-object-method">&nbsp;</a> 
```python
@property
background(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2036)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2036?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.colorbar" class="docs-object-method">&nbsp;</a> 
```python
@property
colorbar(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2043)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2043?message=Update%20Docs)]
</div>


<a id="McUtils.Plots.Graphics.GraphicsGrid.prep_show" class="docs-object-method">&nbsp;</a> 
```python
prep_show(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Graphics/GraphicsGrid.py#L2068)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics/GraphicsGrid.py#L2068?message=Update%20Docs)]
</div>
 </div>
</div>




## Examples
Create a multi-panel figure

<div class="card in-out-block" markdown="1">

```python
grid = np.linspace(0, 2 * np.pi, 100)
grid_2D = np.meshgrid(grid, grid)
x = grid_2D[1]
y = grid_2D[0]

main = GraphicsGrid(ncols=3, nrows=1, theme='Solarize_Light2', figure_label='my beuatufil triptych',
                            padding=((35, 60), (35, 40)), subimage_size=300)
main[0, 0] = ContourPlot(x, y, np.sin(y), plot_label='$sin(x)$',
                         axes_labels=[None, "cats (cc)"],
                         figure=main[0, 0]
                         )
main[0, 1] = ContourPlot(x, y, np.sin(x) * np.cos(y),
                         plot_label='$sin(x)cos(y)$',
                         axes_labels=[Styled("dogs (arb.)", {'color': 'red'}), None],
                         figure=main[0, 1])
main[0, 2] = ContourPlot(x, y, np.cos(y), plot_label='$cos(y)$', figure=main[0, 2])
main.colorbar = {"graphics": main[0, 1].graphics}
```

<div class="card-body out-block" markdown="1">

![plot](../../../img/McUtils_GraphicsGrid_1.png)
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Plots/Graphics/GraphicsGrid.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Plots/Graphics/GraphicsGrid.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Plots/Graphics/GraphicsGrid.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Plots/Graphics/GraphicsGrid.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Graphics.py#L1670?message=Update%20Docs)   
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