## <a id="McUtils.GaussianInterface.GaussianImporter.GaussianLogReader">GaussianLogReader</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianImporter.py#L23)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianImporter.py#L23?message=Update%20Docs)]
</div>

Implements a stream based reader for a Gaussian .log file.
This is inherits from the `FileStreamReader` base, and takes a two pronged approach to getting data.
First, a block is found in a log file based on a pair of tags.
Next, a function (usually based on a `StringParser`) is applied to this data to convert it into a usable data format.
The goal is to move toward wrapping all returned data in a `QuantityArray` so as to include data type information, too.

You can see the full list of available keys in the `GaussianLogComponents` module, but currently they are:
* `"Header"`: the header for the Gaussian job
* `"InputZMatrix"`: the string of the input Z-matrix
* `"CartesianCoordinates"`: all the Cartesian coordinates in the file
* `"ZMatCartesianCoordinates"`: all of the Cartesian coordinate in Z-matrix orientation
* `"StandardCartesianCoordinates"`: all of the Cartesian coordinates in 'standard' orientation
* `"InputCartesianCoordinates"`: all of the Cartesian coordinates in 'input' orientation
* `"ZMatrices"`: all of the Z-matrices
* `"OptimizationParameters"`: all of the optimization parameters
* `"MullikenCharges"`: all of the Mulliken charges
* `"MultipoleMoments"`: all of the multipole moments
* `"DipoleMoments"`: all of the dipole moments
* `"OptimizedDipoleMoments"`: all of the dipole moments from an optimized scan
* `"ScanEnergies"`: the potential surface information from a scan
* `"OptimizedScanEnergies"`: the PES from an optimized scan
* `"XMatrix"`: the anharmonic X-matrix from Gaussian's style of perturbation theory
* `"Footer"`: the footer from a calculation

You can add your own types, too.
If you need something we don't have, give `GaussianLogComponents` a look to see how to add it in.







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse " id="methods" markdown="1">
 ```python
registered_components: OrderedDict
default_keys: tuple
default_ordering: dict
job_default_keys: dict
```
<a id="McUtils.GaussianInterface.GaussianImporter.GaussianLogReader.parse" class="docs-object-method">&nbsp;</a> 
```python
parse(self, keys=None, num=None, reset=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianImporter/GaussianLogReader.py#L58)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianImporter/GaussianLogReader.py#L58?message=Update%20Docs)]
</div>
The main function we'll actually use. Parses bits out of a .log file.
  - `keys`: `str or list(str)`
    > the keys we'd like to read from the log file
  - `num`: `int or None`
    > for keys with multiple entries, the number of entries to pull
  - `:returns`: `dict`
    > t
h
e
 
d
a
t
a
 
p
u
l
l
e
d
 
f
r
o
m
 
t
h
e
 
l
o
g
 
f
i
l
e
,
 
s
t
r
u
n
g
 
t
o
g
e
t
h
e
r
 
a
s
 
a
 
`
d
i
c
t
`
 
a
n
d
 
k
e
y
e
d
 
b
y
 
t
h
e
 
_
k
e
y
s
_


<a id="McUtils.GaussianInterface.GaussianImporter.GaussianLogReader.get_default_keys" class="docs-object-method">&nbsp;</a> 
```python
get_default_keys(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianImporter/GaussianLogReader.py#L107)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianImporter/GaussianLogReader.py#L107?message=Update%20Docs)]
</div>
Tries to get the default keys one might be expected to want depending on the type of job as determined from the Header
Currently only supports 'opt', 'scan', and 'popt' as job types.
  - `:returns`: `tuple(str)`
    > k
e
y
 
l
i
s
t
i
n
g


<a id="McUtils.GaussianInterface.GaussianImporter.GaussianLogReader.read_props" class="docs-object-method">&nbsp;</a> 
```python
read_props(file, keys): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianImporter/GaussianLogReader.py#L136)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianImporter/GaussianLogReader.py#L136?message=Update%20Docs)]
</div>
 </div>
</div>




## Examples
We're working on improving this documentation, but in the meantime, here are the unit tests we use

```python
from McUtils.GaussianInterface import GaussianLogReader

def test_GetLogInfo(self):
    with GaussianLogReader(TestManager.test_data("tbhp_030.log")) as reader:
        parse = reader.parse("Header")
    self.assertIn("P", parse["Header"].job)

def test_DefaultLogParse(self):
    with GaussianLogReader(TestManager.test_data("tbhp_030.log")) as reader:
        parse = reader.parse()
    self.assertLess(parse["OptimizedScanEnergies"][1][0], -308)

def test_GetDipoles(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("DipoleMoments")
    dips = parse["DipoleMoments"]
    self.assertIsInstance(dips, np.ndarray)
    self.assertEquals(dips.shape, (251, 3))

def test_GaussianLoad(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("InputZMatrix")
    zmat = parse["InputZMatrix"]
    self.assertIsInstance(zmat, str)

def test_GaussianAllCartesians(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("CartesianCoordinates")
    carts = parse["CartesianCoordinates"]
    self.assertIsInstance(carts[1], np.ndarray)
    self.assertEquals(carts[1].shape, (502, 3, 3))

def test_GaussianCartesians(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("CartesianCoordinates", num=15)
    carts = parse["CartesianCoordinates"]
    self.assertIsInstance(carts[1], np.ndarray)
    self.assertEquals(carts[1].shape, (15, 3, 3))

def test_GaussianStandardCartesians(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("StandardCartesianCoordinates", num=15)
    carts = parse["StandardCartesianCoordinates"]
    self.assertIsInstance(carts[1], np.ndarray)
    self.assertEquals(carts[1].shape, (15, 3, 3))

def test_GaussianZMatrixCartesians(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("ZMatCartesianCoordinates", num=15)
    carts = parse["ZMatCartesianCoordinates"]
    self.assertIsInstance(carts[1], np.ndarray)
    self.assertEquals(carts[1].shape, (15, 3, 3))

def test_GZMatCoords(self):
    with GaussianLogReader(self.test_log_water) as reader:
        parse = reader.parse("ZMatrices", num = 3)
    zmats = parse["ZMatrices"]
    # print(zmats, file=sys.stderr)
    self.assertIsInstance(zmats[0][1][0], str)
    self.assertIsInstance(zmats[1], np.ndarray)
    self.assertEquals(zmats[1].shape, (3, 3))
    self.assertEquals(zmats[2].shape, (3, 3, 3))

def test_ScanEnergies(self):
    with GaussianLogReader(self.test_log_tet) as reader:
        parse = reader.parse("ScanEnergies", num=3)
    engs = parse["ScanEnergies"]
    self.assertIsInstance(engs, dict)
    self.assertIsInstance(engs["values"], np.ndarray)
    self.assertIsInstance(engs["coords"], np.ndarray)
    self.assertEquals(engs["coords"].shape, (5,))
    self.assertEquals(engs["values"].shape, (30, 5))

def test_GZMatCoordsBiggie(self):
    num_pulled = 5
    num_entries = 8
    with GaussianLogReader(self.test_log_h2) as reader:
        parse = reader.parse("ZMatrices", num = num_pulled)
    zmats = parse["ZMatrices"]
    # print(zmats, file=sys.stderr)
    self.assertIsInstance(zmats[1], np.ndarray)
    self.assertEquals(zmats[1].shape, (num_entries, 3))
    self.assertEquals(zmats[2].shape, (num_pulled, num_entries, 3))

def test_OptScanEnergies(self):
    with GaussianLogReader(self.test_log_opt) as reader:
        parse = reader.parse("OptimizedScanEnergies")
    e, c = parse["OptimizedScanEnergies"]
    self.assertIsInstance(e, np.ndarray)
    self.assertEquals(e.shape, (10,))
    self.assertEquals(len(c.keys()), 14)
    self.assertEquals(list(c.values())[0].shape, (10,))

def test_OptDips(self):
    with GaussianLogReader(self.test_log_tet_rel) as reader:
        parse = reader.parse(["OptimizedScanEnergies", "OptimizedDipoleMoments"])
    c = np.array(parse["OptimizedDipoleMoments"])
    self.assertIsInstance(c, np.ndarray)
    self.assertEquals(c.shape, (30,3))

def test_XMatrix(self):
    file=TestManager.test_data('qooh1.log')
    with GaussianLogReader(file) as reader:
        parse = reader.parse("XMatrix")
    X = np.array(parse["XMatrix"])

    self.assertIsInstance(X, np.ndarray)
    self.assertEquals(X.shape, (39, 39))
    self.assertEquals(X[0, 10], -0.126703)
    self.assertEquals(X[33, 26], -0.642702E-1)

```






---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/GaussianInterface/GaussianImporter/GaussianLogReader.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/GaussianInterface/GaussianImporter/GaussianLogReader.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/GaussianInterface/GaussianImporter/GaussianLogReader.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/GaussianInterface/GaussianImporter/GaussianLogReader.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianImporter.py#L23?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>