## <a id="McUtils.Plots.Plots.ContourPlot">ContourPlot</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L921)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L921?message=Update%20Docs)]
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
method: str
```


 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [GraphicsGrid](#GraphicsGrid)
- [PlotGridStyling](#PlotGridStyling)

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

#### <a name="GraphicsGrid">GraphicsGrid</a>
```python
    def test_GraphicsGrid(self):

        main = GraphicsGrid(ncols=3, nrows=1)
        grid = np.linspace(0, 2 * np.pi, 100)
        grid_2D = np.meshgrid(grid, grid)
        main[0, 0] = ContourPlot(grid_2D[1], grid_2D[0], np.sin(grid_2D[0]), figure=main[0, 0])
        main[0, 1] = ContourPlot(grid_2D[1], grid_2D[0], np.sin(grid_2D[0]) * np.cos(grid_2D[1]), figure=main[0, 1])
        main[0, 2] = ContourPlot(grid_2D[1], grid_2D[0], np.cos(grid_2D[1]), figure=main[0, 2])
        # main.show()

        main.savefig(self.result_file("test_GraphicsGrid.png"))
        main.close()
```
#### <a name="PlotGridStyling">PlotGridStyling</a>
```python
    def test_PlotGridStyling(self):
        main = GraphicsGrid(ncols=3, nrows=1, theme='Solarize_Light2', figure_label='my beuatufil triptych',
                            padding=((35, 60), (35, 40)))
        grid = np.linspace(0, 2 * np.pi, 100)
        grid_2D = np.meshgrid(grid, grid)
        x = grid_2D[1]; y = grid_2D[0]
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

        # main.show()

        main.savefig(self.result_file("test_PlotGridStyling.png"))
        main.close()
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/ContourPlot.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/ContourPlot.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/ContourPlot.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/ContourPlot.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L921?message=Update%20Docs)