## <a id="McUtils.Parsers.RegexPatterns.RegexPattern">RegexPattern</a>
Represents a combinator structure for building more complex regexes

It might be worth working with this combinator structure in a _lazy_ fashion so that we can drill down
into the expression structure... that way we can define a sort-of Regex calculus that we can use to build up higher
order regexes but still be able to recursively inspect subparts?

### Properties and Methods
<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, pat, name=None, children=None, parents=None, dtype=None, repetitions=None, key=None, joiner='', join_function=None, wrapper_function=None, suffix=None, prefix=None, parser=None, handler=None, default_value=None, capturing=None, allow_inner_captures=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L55)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L55?message=Update%20Docs)]
</div>


- `pat`: `str | callable`
    >No description...
- `name`: `str`
    >No description...
- `dtype`: `Any`
    >No description...
- `repetitions`: `Any`
    >No description...
- `key`: `Any`
    >No description...
- `joiner`: `Any`
    >No description...
- `children`: `Any`
    >No description...
- `parents`: `Any`
    >No description...
- `wrapper_function`: `Any`
    >No description...
- `suffix`: `Any`
    >No description...
- `prefix`: `Any`
    >No description...
- `parser`: `Any`
    >No description...
- `handler`: `Any`
    >No description...
- `capturing`: `Any`
    >No description...
- `allow_inner_captures`: `Any`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.pat" class="docs-object-method">&nbsp;</a> 
```python
@property
pat(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.children" class="docs-object-method">&nbsp;</a> 
```python
@property
children(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `tuple[RegexPattern]`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.child_count" class="docs-object-method">&nbsp;</a> 
```python
@property
child_count(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `int`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.child_map" class="docs-object-method">&nbsp;</a> 
```python
@property
child_map(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

Returns the map to subregexes for named regex components
- `:returns`: `Dict[str, RegexPattern]`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.parents" class="docs-object-method">&nbsp;</a> 
```python
@property
parents(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `tuple[RegexPattern]`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.joiner" class="docs-object-method">&nbsp;</a> 
```python
@property
joiner(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `str`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.join_function" class="docs-object-method">&nbsp;</a> 
```python
@property
join_function(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `function`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.suffix" class="docs-object-method">&nbsp;</a> 
```python
@property
suffix(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `str | RegexPattern`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.prefix" class="docs-object-method">&nbsp;</a> 
```python
@property
prefix(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>


- `:returns`: `str | RegexPattern`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.dtype" class="docs-object-method">&nbsp;</a> 
```python
@property
dtype(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

Returns the StructuredType for the matched object

        The basic thing we do is build the type from the contained child dtypes
        The process effectively works like this:
            If there's a single object, we use its dtype no matter what
            Otherwise, we add together our type objects one by one, allowing the StructuredType to handle the calculus

        After we've built our raw types, we compute the shape on top of these, using the assigned repetitions object
        One thing I realize now I failed to do is to include the effects of sub-repetitions... only a single one will
        ever get called.
- `:returns`: `None | StructuredType`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.is_repeating" class="docs-object-method">&nbsp;</a> 
```python
@property
is_repeating(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.capturing" class="docs-object-method">&nbsp;</a> 
```python
@property
capturing(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.get_capturing_groups" class="docs-object-method">&nbsp;</a> 
```python
get_capturing_groups(self, allow_inners=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L329)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L329?message=Update%20Docs)]
</div>

We walk down the tree to find the children with capturing groups in them and
        then find the outermost RegexPattern for those unless allow_inners is on in which case we pull them all

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.captures" class="docs-object-method">&nbsp;</a> 
```python
@property
captures(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

Subtly different from capturing n that it will tell us if we need to use the group in post-processing, essentially
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.capturing_groups" class="docs-object-method">&nbsp;</a> 
```python
@property
capturing_groups(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

Returns the capturing children for the pattern
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.named_groups" class="docs-object-method">&nbsp;</a> 
```python
@property
named_groups(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

Returns the named children for the pattern
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.combine" class="docs-object-method">&nbsp;</a> 
```python
combine(self, other, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L407)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L407?message=Update%20Docs)]
</div>

Combines self and other
- `other`: `RegexPattern | str`
    >No description...
- `:returns`: `str | callable`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.wrap" class="docs-object-method">&nbsp;</a> 
```python
wrap(self, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L429)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L429?message=Update%20Docs)]
</div>

Applies wrapper function

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.build" class="docs-object-method">&nbsp;</a> 
```python
build(self, joiner=None, prefix=None, suffix=None, recompile=True, no_captures=False, verbose=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L443)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L443?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.compiled" class="docs-object-method">&nbsp;</a> 
```python
@property
compiled(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.add_parent" class="docs-object-method">&nbsp;</a> 
```python
add_parent(self, parent): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L528)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L528?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.remove_parent" class="docs-object-method">&nbsp;</a> 
```python
remove_parent(self, parent): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L530)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L530?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.add_child" class="docs-object-method">&nbsp;</a> 
```python
add_child(self, child): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L533)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L533?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.add_children" class="docs-object-method">&nbsp;</a> 
```python
add_children(self, children): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L539)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L539?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.remove_child" class="docs-object-method">&nbsp;</a> 
```python
remove_child(self, child): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L545)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L545?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.insert_child" class="docs-object-method">&nbsp;</a> 
```python
insert_child(self, index, child): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L551)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L551?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.invalidate_cache" class="docs-object-method">&nbsp;</a> 
```python
invalidate_cache(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L556)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L556?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__copy__" class="docs-object-method">&nbsp;</a> 
```python
__copy__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L571)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L571?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__add__" class="docs-object-method">&nbsp;</a> 
```python
__add__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L581)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L581?message=Update%20Docs)]
</div>

Combines self and other
- `other`: `RegexPattern`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__radd__" class="docs-object-method">&nbsp;</a> 
```python
__radd__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L597)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L597?message=Update%20Docs)]
</div>

Combines self and other
- `other`: `RegexPattern`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, other, *args, name=None, dtype=None, repetitions=None, key=None, joiner=None, join_function=None, wrap_function=None, suffix=None, prefix=None, multiline=None, parser=None, handler=None, capturing=None, default=None, allow_inner_captures=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L613)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L613?message=Update%20Docs)]
</div>

Wraps self around other
- `other`: `RegexPattern`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L685)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L685?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__str__" class="docs-object-method">&nbsp;</a> 
```python
__str__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L692)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L692?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L696)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L696?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.match" class="docs-object-method">&nbsp;</a> 
```python
match(self, txt): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L700)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L700?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.search" class="docs-object-method">&nbsp;</a> 
```python
search(self, txt): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L702)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L702?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.findall" class="docs-object-method">&nbsp;</a> 
```python
findall(self, txt): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L704)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L704?message=Update%20Docs)]
</div>

<a id="McUtils.Parsers.RegexPatterns.RegexPattern.finditer" class="docs-object-method">&nbsp;</a> 
```python
finditer(self, txt): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parsers/RegexPatterns.py#L706)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parsers/RegexPatterns.py#L706?message=Update%20Docs)]
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [RegexGroups](#RegexGroups)
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

#### <a name="RegexGroups">RegexGroups</a>
```python
    def test_RegexGroups(self):
        # tests whether we capture subgroups or not (by default _not_)

        test_str = "1 2 3 4 a b c d "
        pattern = RegexPattern(
            (
                Capturing(
                    Repeating(
                        Capturing(Repeating(PositiveInteger, 2, 2, suffix=Optional(Whitespace)))
                    )
                ),
                Repeating(Capturing(ASCIILetter), suffix=Whitespace)
            )
        )
        self.assertEquals(len(pattern.search(test_str).groups()), 2)
```
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

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Parsers/RegexPatterns/RegexPattern.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Parsers/RegexPatterns/RegexPattern.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Parsers/RegexPatterns/RegexPattern.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Parsers/RegexPatterns/RegexPattern.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parsers/RegexPatterns.py?message=Update%20Docs)