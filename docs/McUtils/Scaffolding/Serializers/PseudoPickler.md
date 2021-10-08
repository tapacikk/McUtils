## <a id="McUtils.Scaffolding.Serializers.PseudoPickler">PseudoPickler</a>
A simple plugin to work _like_ pickle, in that it should
hopefully support serializing arbitrary python objects, but which
doesn't attempt to put stuff down to a single `bytearray`, instead
supporting objects with `to_state` and `from_state` methods by converting
them to more primitive serializble types like arrays, strings, numbers,
etc.
Falls back to naive pickling when necessary.

### Properties and Methods
<a id="McUtils.Scaffolding.Serializers.PseudoPickler.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, allow_pickle=False, protocol=1, b64encode=False): 
```

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.to_state" class="docs-object-method">&nbsp;</a>
```python
to_state(self, obj, cache=None): 
```
Tries to extract state from `obj`, first through its `to_state`
        interface, but that failing by recursively walking the object
        tree
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, obj, cache=None): 
```
Serializes an object first by checking for a `to_state`
        method, and that missing, by converting to primitive-ish types
        in a recursive strategy if the object passes `is_simple`, otherwise
        falling back to `pickle`
- `obj`: `Any`
    >object to be serialized
- `:returns`: `dict`
    >spec for the pseudo-pickled data

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, spec): 
```
Deserializes from an object spec, dispatching
        to regular pickle where necessary
- `object`: `Any`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Serializers/PseudoPickler.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Serializers/PseudoPickler.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Serializers/PseudoPickler.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Serializers/PseudoPickler.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Serializers.py?message=Update%20Docs)