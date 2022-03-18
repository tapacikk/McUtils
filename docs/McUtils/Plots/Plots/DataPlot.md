## <a id="McUtils.Plots.Plots.DataPlot">DataPlot</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L386)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L386?message=Update%20Docs)]
</div>

Makes a 2D plot of arbitrary data using a plot method that handles that data type

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Plots.Plots.DataPlot.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *params, plot_style=None, method=None, figure=None, axes=None, subplot_kw=None, colorbar=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L390)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L390?message=Update%20Docs)]
</div>


- `params`: `Any`
    >_empty_ or _data_
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
- `opts`: `Any`
    >options to be fed in when initializing the Graphics

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/DataPlot.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/DataPlot.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/DataPlot.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/DataPlot.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L386?message=Update%20Docs)