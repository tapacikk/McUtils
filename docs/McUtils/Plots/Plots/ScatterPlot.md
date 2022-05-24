## <a id="McUtils.Plots.Plots.ScatterPlot">ScatterPlot</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L413)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L413?message=Update%20Docs)]
</div>

Inherits from `Plot`.
Plots a bunch of x values against a bunch of y values using the `scatter` method.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
known_styles: set
style_mapping: dict
method: str
```


 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [Scatter](#Scatter)

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

#### <a name="Scatter">Scatter</a>
```python
    def test_Scatter(self):
        pts = np.random.rand(50, 2)
        plot = ScatterPlot(*pts.T,
                           aspect_ratio=2,
                           image_size=250
                           )
        # plot.show()
        plot.savefig(self.result_file("test_Scatter.pdf"), format='pdf')
        plot.close()
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/ScatterPlot.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/ScatterPlot.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/ScatterPlot.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/ScatterPlot.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L413?message=Update%20Docs)