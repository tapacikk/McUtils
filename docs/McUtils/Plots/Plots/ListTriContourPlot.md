## <a id="McUtils.Plots.Plots.ListTriContourPlot">ListTriContourPlot</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L655)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L655?message=Update%20Docs)]
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Plots.Plots.ListTriContourPlot.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, griddata, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Plots.py#L656)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L656?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [ListTriContourPlot](#ListTriContourPlot)
- [Animation](#Animation)

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

#### <a name="ListTriContourPlot">ListTriContourPlot</a>
```python
    def test_ListTriContourPlot(self):
        pts = np.pi*np.random.rand(150, 2)
        sins = np.sin(pts[:, 0])
        coses = np.cos(pts[:, 1])
        ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
        plot = ListTriContourPlot(ptss)
        plot.add_colorbar()

        plot.savefig(self.result_file("test_ListTriContourPlot.png"))
        plot.close()
```
#### <a name="Animation">Animation</a>
```python
    def test_Animation(self):
        "Currently broken"
        def get_data(*args):
            pts = np.pi*np.random.normal(scale = .25, size=(10550, 2))
            sins = np.sin(pts[:, 0])
            coses = np.cos(pts[:, 1])
            ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
            return (ptss, )
        plot = ListTriContourPlot(*get_data(),
                                  animate = get_data,
                                  plot_range = [
                                      [-np.pi, np.pi],
                                      [-np.pi, np.pi]
                                  ]
                                  )

        plot.show()

        plot.savefig(self.result_file("test_ListTriContourPlot.gif"))
        plot.close()
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Plots/ListTriContourPlot.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Plots/ListTriContourPlot.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Plots/ListTriContourPlot.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Plots/ListTriContourPlot.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Plots.py#L655?message=Update%20Docs)