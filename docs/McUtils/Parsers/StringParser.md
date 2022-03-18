# <a id="McUtils.Parsers.StringParser">McUtils.Parsers.StringParser</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/StringParser)]
</div>
    


<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[StringParser](StringParser/StringParser.md)   
</div>
   <div class="col" markdown="1">
[StringParserException](StringParser/StringParserException.md)   
</div>
</div>
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [OptScan](#OptScan)
- [BasicParse](#BasicParse)

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
from McUtils.Parsers import *
import sys, os, numpy as np
```

All tests are wrapped in a test class
```python
class ParserTests(TestCase):
```

 </div>
</div>

#### <a name="OptScan">OptScan</a>
```python
    def test_OptScan(self):

        eigsPattern = RegexPattern(
            (
                "Eigenvalues --",
                Repeating(Capturing(Number), suffix=Optional(Whitespace))
            ),
            joiner=Whitespace
        )

        coordsPattern = RegexPattern(
            (
                Capturing(VariableName),
                Repeating(Capturing(Number), suffix=Optional(Whitespace))
            ),
            prefix=Whitespace,
            joiner=Whitespace
        )

        full_pattern = RegexPattern(
            (
                Named(eigsPattern,
                      "Eigenvalues"
                      #parser=lambda t: np.array(Number.findall(t), 'float')
                      ),
                Named(Repeating(coordsPattern, suffix=Optional(Newline)), "Coordinates")
            ),
            joiner=Newline
        )

        with open(TestManager.test_data('scan_params_test.txt')) as test:
            test_str = test.read()

        parser = StringParser(full_pattern)
        parse_res = parser.parse_all(test_str)
        parse_single = parser.parse(test_str)
        parse_its = list(parser.parse_iter(test_str))

        self.assertEquals(parse_res.shape, [(4, 5), [(4, 32), (4, 32, 5)]])
        self.assertIsInstance(parse_res["Coordinates"][1].array, np.ndarray)
        self.assertEquals(int(parse_res["Coordinates"][1, 0].sum()), 3230)
```
#### <a name="BasicParse">BasicParse</a>
```python
    def test_BasicParse(self):
        regex = RegexPattern(
            (
                Named(PositiveInteger, "NumAtoms"),
                Named(
                    Repeating(Any, min = None), "Comment", dtype=str
                ),
                Named(
                    Repeating(
                        Capturing(
                            Repeating(Capturing(Number), 3, 3, prefix = Whitespace, suffix = Optional(Whitespace)),
                            handler= StringParser.array_handler(shape = (None, 3))
                        ),
                        suffix = Optional(Newline)
                    ),
                    "Atoms"
                )
            ),
            "XYZ",
            joiner=Newline
        )

        with open(TestManager.test_data('coord_parse.txt')) as test:
            test_str = test.read()

        res = StringParser(regex).parse(test_str)

        comment_string = res["Comment"].array[0]
        self.assertTrue('comment' in comment_string)
        self.assertEquals(res['Atoms'].array.shape, (4, 3))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Parsers/StringParser.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Parsers/StringParser.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Parsers/StringParser.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Parsers/StringParser.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/StringParser/__init__.py?message=Update%20Docs)