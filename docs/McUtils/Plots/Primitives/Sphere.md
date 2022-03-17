## <a id="McUtils.Plots.Primitives.Sphere">Sphere</a>


### Properties and Methods
<a id="McUtils.Plots.Primitives.Sphere.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, position=(0, 0, 0), radius=1, sphere_points=48, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Primitives.py#L93)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Primitives.py#L93?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Primitives.Sphere.plot" class="docs-object-method">&nbsp;</a> 
```python
plot(self, graphics, *args, sphere_points=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Primitives.py#L99)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Primitives.py#L99?message=Update%20Docs)]
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [VTK](#VTK)

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

#### <a name="VTK">VTK</a>
```python
    def test_VTK(self):
        plot = Graphics3D(backend="VTK", image_size=[1500, 500])
        Sphere().plot(plot)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Plots/Primitives/Sphere.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Plots/Primitives/Sphere.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Plots/Primitives/Sphere.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Plots/Primitives/Sphere.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Plots/Primitives.py?message=Update%20Docs)