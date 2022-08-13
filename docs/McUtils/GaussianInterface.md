# <a id="McUtils.GaussianInterface">McUtils.GaussianInterface</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/tree/master/GaussianInterface)]
</div>
    
A module for making use of the results of calculations run by the Gaussian electronic structure package.
We'd like to be able to also support the NWChem and Psi4 packages, but haven't had the time, yet, to write it out.

Two main avenues of support are provided:
    1. importing Gaussian results
    2. setting up Gaussian jobs

The first is likely to be more useful to you, but we're hoping to be able to hook (2.) into the `Psience.Molecools` package.
The goal there is to provide automated support for setting up scans of molecular vibrations & the like.

There are already direct hooks into (1.) in `Psience.Data` through the `DipoleSurface` and `PotentialSurface` objects.
These are still in the prototype stage, but hopefully will allow us to unify strands of our Gaussian support,
 and also make it easy to unify support for Psi4 and NWChem data, once we have the basic interface down.

<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[GaussianFChkReader](GaussianInterface/GaussianImporter/GaussianFChkReader.md)   
</div>
   <div class="col" markdown="1">
[GaussianLogReader](GaussianInterface/GaussianImporter/GaussianLogReader.md)   
</div>
   <div class="col" markdown="1">
[GaussianLogReaderException](GaussianInterface/GaussianImporter/GaussianLogReaderException.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[GaussianFChkReaderException](GaussianInterface/GaussianImporter/GaussianFChkReaderException.md)   
</div>
   <div class="col" markdown="1">
[GaussianJob](GaussianInterface/GaussianJob/GaussianJob.md)   
</div>
   <div class="col" markdown="1">
[GaussianJobArray](GaussianInterface/GaussianJob/GaussianJobArray.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[FchkForceConstants](GaussianInterface/FChkDerivatives/FchkForceConstants.md)   
</div>
   <div class="col" markdown="1">
[FchkForceDerivatives](GaussianInterface/FChkDerivatives/FchkForceDerivatives.md)   
</div>
   <div class="col" markdown="1">
[FchkDipoleDerivatives](GaussianInterface/FChkDerivatives/FchkDipoleDerivatives.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[FchkDipoleHigherDerivatives](GaussianInterface/FChkDerivatives/FchkDipoleHigherDerivatives.md)   
</div>
   <div class="col" markdown="1">
[FchkDipoleNumDerivatives](GaussianInterface/FChkDerivatives/FchkDipoleNumDerivatives.md)   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>



## Examples

## FChk Parsing

Gaussian `.fchk` files have a set structure which looks roughly like
```lang-none
key    data_type     data_size
 data
```
This allows us to provide a complete parser for any `key`.
The actual parser is a subclass of `[Parsers.FileStreamReader]`(../Parsers/) called `GaussianFchkReader`.

The syntax to parse is straightforward

```python
target_keys = {"Current cartesian coordinates", "Numerical dipole derivatives"}
with GaussianFchkReader("/path/to/output.log") as parser:
    res = parser.parse(target_keys)
```

and to access properties you will pull them from the dict, `res`

```python
my_coords = res["Current cartesian coordinates"]
```


## Log Parsing

Gaussian `.log` files are totally unstructured (and a bit of a disaster). 
This means we need to write custom parsing logic for every field we might want.
The basic supported formats are defined in `GaussianLogComponents.py`. 
The actual parser is a subclass of [`Parsers.FileStreamReader`](../Parsers/) called `GaussianLogReader`.

The syntax to parse is straightforward

```python
target_keys = {"StandardCartesianCoordinates", "DipoleMoments"}
with GaussianLogReader("/path/to/output.log") as parser:
    res = parser.parse(target_keys)
```

and to access properties you will pull them from the dict, `res`

```python
my_coords = res["StandardCartesianCoordinates"]
```

### Adding New Parsing Fields

New parse fields can be added by registering a property on `GaussianLogComponents`. 
Each field is defined as a dict like

```python
GaussianLogComponents["Name"] = {
    "description" : string, # used for docmenting what we have
    "tag_start"   : start_tag, # starting delimeter for a block
    "tag_end"     : end_tag, # ending delimiter for a block None means apply the parser upon tag_start
    "parser"      : parser, # function that'll parse the returned list of blocks (for "List") or block (for "Single")
    "mode"        : mode # "List" or "Single"
}
```

The `mode` argument specifies whether all blocks should be matched first and send to the `parser` (`"List"`) or if they should be fed in one-by-one `"Single"`.
This often provides a tradeoff between parsing efficiency and memory efficiency.

The `parser` can be any function, but commonly is built off of a [`Parsers.StringParser`](../Parsers/). 
See the documentation for `StringParser` for more.

You can add to `GaussianLogComponents` at runtime.
Not all changes need to be integrated directly into the file.
{: .alert .alert-info}

## GJF Setup

Support is also provided for the automatic generation of Gaussian job files (`.gjf`) through the `GaussianJob` class.


<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [GetLogInfo](#GetLogInfo)
- [DefaultLogParse](#DefaultLogParse)
- [GetDipoles](#GetDipoles)
- [GaussianLoad](#GaussianLoad)
- [GaussianAllCartesians](#GaussianAllCartesians)
- [GaussianCartesians](#GaussianCartesians)
- [GaussianStandardCartesians](#GaussianStandardCartesians)
- [GaussianZMatrixCartesians](#GaussianZMatrixCartesians)
- [GZMatCoords](#GZMatCoords)
- [ScanEnergies](#ScanEnergies)
- [OptScanEnergies](#OptScanEnergies)
- [OptDips](#OptDips)
- [XMatrix](#XMatrix)
- [Fchk](#Fchk)
- [ForceConstants](#ForceConstants)
- [ForceThirdDerivatives](#ForceThirdDerivatives)
- [ForceFourthDerivatives](#ForceFourthDerivatives)
- [FchkMasses](#FchkMasses)

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
from McUtils.GaussianInterface import *
import sys, os, numpy as np
```

All tests are wrapped in a test class
```python
class GaussianInterfaceTests(TestCase):
    def setUp(self):
        self.test_log_water = TestManager.test_data("water_OH_scan.log")
        self.test_log_freq = TestManager.test_data("water_freq.log")
        self.test_log_opt = TestManager.test_data("water_dimer_test.log")
        self.test_fchk = TestManager.test_data("water_freq.fchk")
        self.test_log_h2 = TestManager.test_data("outer_H2_scan_new.log")
        self.test_scan = TestManager.test_data("water_OH_scan.log")
        self.test_rel_scan = TestManager.test_data("tbhp_030.log")
```

 </div>
</div>

#### <a name="GetLogInfo">GetLogInfo</a>
```python
    def test_GetLogInfo(self):
        with GaussianLogReader(self.test_rel_scan) as reader:
            parse = reader.parse("Header")
        self.assertIn("P", parse["Header"].job)
```
#### <a name="DefaultLogParse">DefaultLogParse</a>
```python
    def test_DefaultLogParse(self):
        with GaussianLogReader(self.test_rel_scan) as reader:
            parse = reader.parse()
        self.assertLess(parse["OptimizedScanEnergies"][0][1], -308)
```
#### <a name="GetDipoles">GetDipoles</a>
```python
    def test_GetDipoles(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("DipoleMoments")
        dips = parse["DipoleMoments"]
        self.assertIsInstance(dips, np.ndarray)
        self.assertEquals(dips.shape, (251, 3))
```
#### <a name="GaussianLoad">GaussianLoad</a>
```python
    def test_GaussianLoad(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("InputZMatrix")
        zmat = parse["InputZMatrix"]
        self.assertIsInstance(zmat, str)
```
#### <a name="GaussianAllCartesians">GaussianAllCartesians</a>
```python
    def test_GaussianAllCartesians(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("CartesianCoordinates")
        carts = parse["CartesianCoordinates"]
        self.assertIsInstance(carts[1], np.ndarray)
        self.assertEquals(carts[1].shape, (502, 3, 3))
```
#### <a name="GaussianCartesians">GaussianCartesians</a>
```python
    def test_GaussianCartesians(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("CartesianCoordinates", num=15)
        carts = parse["CartesianCoordinates"]
        self.assertIsInstance(carts[1], np.ndarray)
        self.assertEquals(carts[1].shape, (15, 3, 3))
```
#### <a name="GaussianStandardCartesians">GaussianStandardCartesians</a>
```python
    def test_GaussianStandardCartesians(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("StandardCartesianCoordinates", num=15)
        carts = parse["StandardCartesianCoordinates"]
        self.assertIsInstance(carts[1], np.ndarray)
        self.assertEquals(carts[1].shape, (15, 3, 3))
```
#### <a name="GaussianZMatrixCartesians">GaussianZMatrixCartesians</a>
```python
    def test_GaussianZMatrixCartesians(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("ZMatCartesianCoordinates", num=15)
        carts = parse["ZMatCartesianCoordinates"]
        self.assertIsInstance(carts[1], np.ndarray)
        self.assertEquals(carts[1].shape, (15, 3, 3))
```
#### <a name="GZMatCoords">GZMatCoords</a>
```python
    def test_GZMatCoords(self):
        with GaussianLogReader(self.test_log_water) as reader:
            parse = reader.parse("ZMatrices", num = 3)
        zmats = parse["ZMatrices"]
        # print(zmats, file=sys.stderr)
        self.assertIsInstance(zmats[0][1][0], str)
        self.assertIsInstance(zmats[1], np.ndarray)
        self.assertEquals(zmats[1].shape, (3, 3))
        self.assertEquals(zmats[2].shape, (3, 3, 3))
```
#### <a name="ScanEnergies">ScanEnergies</a>
```python
    def test_ScanEnergies(self):
        with GaussianLogReader(self.test_scan) as reader:
            parse = reader.parse("ScanEnergies", num=3)
        engs = parse["ScanEnergies"]
        self.assertIsInstance(engs.energies, np.ndarray)
        self.assertIsInstance(engs.coords, np.ndarray)
        self.assertEquals(engs.coords.shape, (4,))
        self.assertEquals(engs.energies.shape, (251, 4))
```
#### <a name="OptScanEnergies">OptScanEnergies</a>
```python
    def test_OptScanEnergies(self):
        with GaussianLogReader(self.test_log_opt) as reader:
            parse = reader.parse("OptimizedScanEnergies")
        e, c = parse["OptimizedScanEnergies"]
        self.assertIsInstance(e, np.ndarray)
        self.assertEquals(e.shape, (9,))
        self.assertEquals(len(c.keys()), 14)
        self.assertEquals(list(c.values())[0].shape, (9,))
```
#### <a name="OptDips">OptDips</a>
```python
    def test_OptDips(self):
        with GaussianLogReader(self.test_rel_scan) as reader:
            parse = reader.parse(["OptimizedScanEnergies", "OptimizedDipoleMoments"])
        c = np.array(parse["OptimizedDipoleMoments"])
        self.assertIsInstance(c, np.ndarray)
        self.assertEquals(c.shape, (28, 3))
```
#### <a name="XMatrix">XMatrix</a>
```python
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
#### <a name="Fchk">Fchk</a>
```python
    def test_Fchk(self):
        with GaussianFChkReader(self.test_fchk) as reader:
            parse = reader.parse()
        key = next(iter(parse.keys()))
        self.assertIsInstance(key, str)
```
#### <a name="ForceConstants">ForceConstants</a>
```python
    def test_ForceConstants(self):
        n = 3 # water
        with GaussianFChkReader(self.test_fchk) as reader:
            parse = reader.parse("ForceConstants")
        fcs = parse["ForceConstants"]
        self.assertEquals(fcs.n, n)
        self.assertEquals(fcs.array.shape, (3*n, 3*n))
```
#### <a name="ForceThirdDerivatives">ForceThirdDerivatives</a>
```python
    def test_ForceThirdDerivatives(self):
        n = 3 # water
        with GaussianFChkReader(self.test_fchk) as reader:
            parse = reader.parse("ForceDerivatives")
        fcs = parse["ForceDerivatives"]
        tds = fcs.third_deriv_array
        self.assertEquals(fcs.n, n)
        self.assertEquals(tds.shape, ((3*n-6), (3*n), 3*n))
        a = tds[0]
        self.assertTrue(
            np.allclose(tds[0], tds[0].T, rtol=1e-08, atol=1e-08)
        )
        self.assertTrue(
            np.allclose(tds[1], tds[1].T, rtol=1e-08, atol=1e-08)
        )
        self.assertTrue(
            np.allclose(tds[2], tds[2].T, rtol=1e-08, atol=1e-08)
        )
```
#### <a name="ForceFourthDerivatives">ForceFourthDerivatives</a>
```python
    def test_ForceFourthDerivatives(self):
        n = 3 # water
        with GaussianFChkReader(self.test_fchk) as reader:
            parse = reader.parse("ForceDerivatives")
        fcs = parse["ForceDerivatives"]
        tds = fcs.fourth_deriv_array
        self.assertEquals(fcs.n, n)
        self.assertEquals(tds.shape, ((3*n-6), (3*n-6), (3*n), 3*n)) # it's a SparseTensor now
        slice_0 = tds[0, 0].toarray()
        slice_1 = tds[1, 1].toarray()
        slice_2 = tds[2, 2].toarray()
        self.assertTrue(
            np.allclose(slice_0, slice_0.T, rtol=1e-08, atol=1e-08)
        )
        self.assertTrue(
            np.allclose(slice_1, slice_1.T, rtol=1e-08, atol=1e-08)
        )
        self.assertTrue(
            np.allclose(slice_2, slice_2.T, rtol=1e-08, atol=1e-08)
        )
```
#### <a name="FchkMasses">FchkMasses</a>
```python
    def test_FchkMasses(self):
        n = 3 # water
        with GaussianFChkReader(self.test_fchk) as reader:
            parse = reader.parse("AtomicMasses")
        masses = parse["AtomicMasses"]
        self.assertEquals(len(masses), n)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/GaussianInterface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/GaussianInterface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/GaussianInterface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/GaussianInterface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/__init__.py?message=Update%20Docs)