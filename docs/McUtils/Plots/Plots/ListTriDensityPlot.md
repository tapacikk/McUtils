## <a id="McUtils.Plots.Plots.ListTriDensityPlot">ListTriDensityPlot</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L660)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L660?message=Update%20Docs)]
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Plots.Plots.ListTriDensityPlot.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, griddata, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L661)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L661?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [ListTriDensityPlot](#ListTriDensityPlot)

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

#### <a name="ListTriDensityPlot">ListTriDensityPlot</a>
```python
    def test_ListTriDensityPlot(self):
        pts = np.pi*np.random.rand(150, 2)
        sins = np.sin(pts[:, 0])
        coses = np.cos(pts[:, 1])
        ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
        plot = ListTriDensityPlot(ptss)

        plot.savefig(self.result_file("test_ListTriDensityPlot.png"))
        plot.close()
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/ListTriDensityPlot.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/ListTriDensityPlot.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/ListTriDensityPlot.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/ListTriDensityPlot.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L660?message=Update%20Docs)