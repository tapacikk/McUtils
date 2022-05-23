## <a id="McUtils.Plots.Plots.ListPlot3D">ListPlot3D</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1141)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1141?message=Update%20Docs)]
</div>

Convenience 3D plotting class that handles the interpolation first

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Plots.Plots.ListPlot3D.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *params, plot_style=None, method='contour', colorbar=None, figure=None, axes=None, subplot_kw=None, interpolate=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1145)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1145?message=Update%20Docs)]
</div>


- `params`: `Any`
    >either _empty_ or and array of (_x_, _y_, _z_) points
- `plot_style`: `dict | None`
    >the plot styling options to be fed into the plot method
- `method`: `str`
    >the method name as a string
- `figure`: `Graphics | None`
    >the Graphics object on which to plot (None means make a new one)
- `axes`: `None`
    >the axes on which to plot (used in constructing a Graphics, None means make a new one)
- `subplot_kw`: `dict | None`
    >the keywords to pass on when initializing the plot
- `colorbar`: `None | bool | dict`
    >whether to use a colorbar or what options to pass to the colorbar
- `interpolate`: `bool`
    >whether to interpolate the data or not
- `opts`: `Any`
    >options to be fed in when initializing the Graphics

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/ListPlot3D.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/ListPlot3D.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/ListPlot3D.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/ListPlot3D.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1141?message=Update%20Docs)