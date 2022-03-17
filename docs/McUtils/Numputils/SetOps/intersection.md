# <a id="McUtils.Numputils.SetOps.intersection">intersection</a>

```python
intersection(ar1, ar2, assume_unique=False, return_indices=False, sortings=None, union_sorting=None, minimal_dtype=False): 
```
 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [SetOps](#SetOps)

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
from Peeves import BlockProfiler
from McUtils.Numputils import *
from McUtils.Zachary import FiniteDifferenceDerivative
from unittest import TestCase
import numpy as np, functools as ft
```

All tests are wrapped in a test class
```python
class NumputilsTests(TestCase):
    problem_coords = np.array([
                                  [-1.86403557e-17, -7.60465240e-02,  4.62443228e-02],
                                  [ 6.70904773e-17, -7.60465240e-02, -9.53755677e-01],
                                  [ 9.29682337e-01,  2.92315732e-01,  4.62443228e-02],
                                  [ 2.46519033e-32, -1.38777878e-17,  2.25076602e-01],
                                  [-1.97215226e-31,  1.43714410e+00, -9.00306410e-01],
                                  [-1.75999392e-16, -1.43714410e+00, -9.00306410e-01]
    ])
```

 </div>
</div>

#### <a name="SetOps">SetOps</a>
```python
    def test_SetOps(self):

        unums, sorting = unique([1, 2, 3, 4, 5])
        self.assertEquals(unums.tolist(), [1, 2, 3, 4, 5])
        self.assertEquals(sorting.tolist(), [0, 1, 2, 3, 4])

        unums, sorting = unique([1, 1, 3, 4, 5])
        self.assertEquals(unums.tolist(), [1, 3, 4, 5])
        self.assertEquals(sorting.tolist(), [0, 1, 2, 3, 4])

        unums, sorting = unique([1, 3, 1, 1, 1])
        self.assertEquals(unums.tolist(), [1, 3])
        self.assertEquals(sorting.tolist(), [0, 2, 3, 4, 1])

        unums, sorting = unique([[1, 3], [1, 1], [1, 3]])
        self.assertEquals(unums.tolist(), [[1, 1], [1, 3]])
        self.assertEquals(sorting.tolist(), [1, 0, 2])

        inters, sortings, merge = intersection(
            [1, 1, 3, 2, 5],
            [0, 0, 0, 5, 1]
        )
        self.assertEquals(inters.tolist(), [1, 5])
        self.assertEquals(sortings[0].tolist(), [0, 1, 3, 2, 4])
        self.assertEquals(sortings[1].tolist(), [0, 1, 2, 4, 3])

        inters, sortings, merge = intersection(
            [
                [1, 3], [1, 1]
            ],
            [
                [1, 3], [0, 0]
            ]
        )
        self.assertEquals(inters.tolist(), [[1, 3]])
        self.assertEquals(sortings[0].tolist(), [1, 0])
        self.assertEquals(sortings[1].tolist(), [1, 0])

        diffs, sortings, merge = difference(
            [1, 1, 3, 2, 5],
            [0, 0, 0, 5, 1]
        )
        self.assertEquals(diffs.tolist(), [2, 3])
        self.assertEquals(sortings[0].tolist(), [0, 1, 3, 2, 4])
        self.assertEquals(sortings[1].tolist(), [0, 1, 2, 4, 3])

        diffs, sortings, merge = contained(
            [1, 1, 3, 2, 5],
            [0, 0, 0, 5, 1]
        )
        self.assertEquals(diffs.tolist(), [True, True, False, False, True])

        ugh = np.arange(1000)
        bleh = np.random.choice(1000, size=100)
        diffs, sortings, merge = contained(
            bleh,
            ugh
        )
        self.assertEquals(diffs.tolist(), np.isin(bleh, ugh).tolist())

        diffs2, sortings, merge = contained(
            bleh,
            ugh,
            method='find'
        )

        self.assertEquals(diffs.tolist(), diffs2.tolist())
```

 </div>
</div>
___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/SetOps/intersection.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/SetOps/intersection.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/SetOps/intersection.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/SetOps/intersection.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/SetOps.py?message=Update%20Docs)