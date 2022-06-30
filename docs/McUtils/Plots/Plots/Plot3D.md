## <a id="McUtils.Plots.Plots.Plot3D">Plot3D</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1029)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1029?message=Update%20Docs)]
</div>

A base class for 3D plots

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_plot_style: dict
style_mapping: dict
known_styles: set
method: str
plot_classes: dict
```
<a id="McUtils.Plots.Plots.Plot3D.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *params, plot_style=None, method=None, colorbar=None, figure=None, axes=None, subplot_kw=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1036)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1036?message=Update%20Docs)]
</div>


- `params`: `Any`
    >either _empty_ or _x_, _y_, _z_ arrays or _function_, _xrange_, _yrange_
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

<a id="McUtils.Plots.Plots.Plot3D.plot" class="docs-object-method">&nbsp;</a> 
```python
plot(self, *params, **plot_style): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1105)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1105?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Plots.Plot3D.add_colorbar" class="docs-object-method">&nbsp;</a> 
```python
add_colorbar(self, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1111)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1111?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Plots.Plot3D.resolve_method" class="docs-object-method">&nbsp;</a> 
```python
resolve_method(mpl_name): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1119)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1119?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Plots.Plot3D.register" class="docs-object-method">&nbsp;</a> 
```python
register(plot_class): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L1125)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1125?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [Plot3D](#Plot3D)
- [Plot3DDelayed](#Plot3DDelayed)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Plots import *
import sys, os, numpy as np
```

All tests are wrapped in a test class
```python
class PlotsTests(TestCase):
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
    def result_file(self, fname):
        if not os.path.isdir(os.path.join(TestManager.test_dir, "test_results")):
            os.mkdir(os.path.join(TestManager.test_dir, "test_results"))
        return os.path.join(TestManager.test_dir, "test_results", fname)
```

 </div>
</div>

#### <a name="Plot3D">Plot3D</a>
```python
    def test_Plot3D(self):
        import matplotlib.cm as colormaps

        f = lambda pt: np.sin(pt[0]) + np.cos(pt[1])
        plot = Plot3D(f, np.arange(0, 2 * np.pi, .1), np.arange(0, 2 * np.pi, .1),
                      plot_style={
                          "cmap": colormaps.get_cmap('viridis')
                      },
                      axes_labels=['dogs', 'cats',
                                   Styled('rats', color='red')
                                   ],
                      plot_label='my super cool 3D plot',
                      plot_range=[(-5, 5)] * 3,
                      plot_legend='i lik turtle',
                      colorbar=True
                      )
        plot.savefig(self.result_file("test_Plot3D.png"))
        plot.close()
```
#### <a name="Plot3DDelayed">Plot3DDelayed</a>
```python
    def test_Plot3DDelayed(self):
        p = Plot3D(background = 'black')
        for i, c in enumerate(('red', 'white', 'blue')):
            p.plot(
                lambda g: (
                    np.sin(g.T[0]) + np.cos(g.T[1])
                ),
                [-2 + 4/3*i, -2 + 4/3*(i+1)],
                [-2 + 4/3*i, -2 + 4/3*(i+1)],
                color = c)
        # p.show()

        p.savefig(self.result_file("test_Plot3DDelayed.gif"))
        p.close()
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/Plot3D.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/Plot3D.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/Plot3D.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/Plot3D.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L1029?message=Update%20Docs)