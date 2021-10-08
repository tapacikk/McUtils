# <a id="McUtils.Data">McUtils.Data</a>
    
Provides a small data framework for wrapping up datasets into classes for access and loading

### Members

  - [DataHandler](Data/CommonData/DataHandler.md)
  - [DataError](Data/CommonData/DataError.md)
  - [DataRecord](Data/CommonData/DataRecord.md)
  - [AtomData](Data/AtomData/AtomData.md)
  - [AtomDataHandler](Data/AtomData/AtomDataHandler.md)
  - [UnitsData](Data/ConstantsData/UnitsData.md)
  - [UnitsDataHandler](Data/ConstantsData/UnitsDataHandler.md)
  - [BondData](Data/BondData/BondData.md)
  - [BondDataHandler](Data/BondData/BondDataHandler.md)
  - [WavefunctionData](Data/WavefunctionData/WavefunctionData.md)
  - [PotentialData](Data/PotentialData/PotentialData.md)

### Examples

We can work with atomic data. The key can be specified in multiple different ways.

```python
from McUtils.Data import AtomData

assert isinstance(AtomData["H"], dict)
assert isinstance(AtomData["Hydrogen"], dict)
assert isinstance(AtomData["Helium3"], dict)
assert AtomData["Hydrogen2"] is AtomData["Deuterium"]
assert AtomData["H2"] is AtomData["Deuterium"]
assert AtomData["H1"] is AtomData["Hydrogen"]
assert AtomData[8] is AtomData["Oxygen"]
```

A fun property of isotopes

```python
from McUtils.Data import AtomData

assert AtomData["Helium3", "Mass"] < AtomData["T"]["Mass"]
```

We can work with unit conversions. Inverse units are supplied using `"Inverse..."`, prefixes can modify units (e.g. `"Centi"`).

```python
from McUtils.Data import UnitsData

assert UnitsData.data[("Hartrees", "InverseMeters")]["Value"] > 21947463.13
assert UnitsData.data[("Hartrees", "InverseMeters")]["Value"] == UnitsData.convert("Hartrees", "InverseMeters")
assert UnitsData.convert("Hartrees", "Wavenumbers") == UnitsData.convert("Hartrees", "InverseMeters") / 100
assert UnitsData.convert("Hartrees", "Wavenumbers") == UnitsData.convert("Centihartrees", "InverseMeters")
```

Atomic units, as a general system, are supported

```python
from McUtils.Data import UnitsData

assert UnitsData.convert("AtomicMassUnits", "AtomicUnitOfMass") == UnitsData.convert("AtomicMassUnits", "ElectronMass")
assert UnitsData.convert("Wavenumbers", "AtomicUnitOfEnergy") == UnitsData.convert("Wavenumbers", "Hartrees")
```




### Unit Tests

```python

from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Data import *

class DataTests(TestCase):

    @validationTest
    def test_AtomData(self):
        self.assertIsInstance(AtomData["H"], DataRecord)
        self.assertIsInstance(AtomData["Hydrogen"], DataRecord)
        self.assertIsInstance(AtomData["Helium3"], DataRecord)
        self.assertIs(AtomData["Hydrogen2"], AtomData["Deuterium"])
        self.assertIs(AtomData["H2"], AtomData["Deuterium"])
        self.assertIs(AtomData["H1"], AtomData["Hydrogen"])
        self.assertIs(AtomData[8], AtomData["Oxygen"])

    @validationTest
    def test_AtomMasses(self):
        self.assertLess(AtomData["Helium3", "Mass"], AtomData["T"]["Mass"]) # fun weird divergence

    @validationTest
    def test_Conversions(self):
        # print(AtomData["T"]["Mass"]-AtomData["Helium3", "Mass"], file=sys.stderr)
        self.assertGreater(UnitsData.data[("Hartrees", "InverseMeters")]["Value"], 21947463.13)
        self.assertLess(UnitsData.data[("Hartrees", "InverseMeters")]["Value"], 21947463.14)
        self.assertAlmostEqual(
            UnitsData.convert("Hartrees", "Wavenumbers"),
            UnitsData.convert("Hartrees", "InverseMeters") / 100
        )
        self.assertAlmostEqual(
            UnitsData.convert("Hartrees", "Wavenumbers"),
            UnitsData.convert("Centihartrees", "InverseMeters")
        )

    @validationTest
    def test_AtomicUnits(self):
        # print(UnitsData["AtomicUnitOfMass"])
        self.assertAlmostEqual(UnitsData.convert("AtomicMassUnits", "AtomicUnitOfMass"), 1822.888486217313)

    @validationTest
    def test_BondData(self):
        self.assertIsInstance(BondData["H"], dict)
        self.assertLess(BondData["H", "H", 1], 1)
        self.assertLess(BondData["H", "O", 1], 1)
        self.assertGreater(BondData["H", "C", 1], 1)
```

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Data.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Data.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Data.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Data.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Data/__init__.py?message=Update%20Docs)