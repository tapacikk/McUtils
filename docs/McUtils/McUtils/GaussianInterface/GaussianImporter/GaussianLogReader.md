## <a id="McUtils.McUtils.GaussianInterface.GaussianImporter.GaussianLogReader">GaussianLogReader</a>
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

### Properties and Methods
```python
registered_components: OrderedDict
default_keys: tuple
default_ordering: dict
job_default_keys: dict
```
<a id="McUtils.McUtils.GaussianInterface.GaussianImporter.GaussianLogReader.parse" class="docs-object-method">&nbsp;</a>
```python
parse(self, keys=None, num=None, reset=False): 
```
The main function we'll actually use. Parses bits out of a .log file.
- `keys`: `str or list(str)`
    >the keys we'd like to read from the log file
- `num`: `int or None`
    >for keys with multiple entries, the number of entries to pull
- `:returns`: `dict`
    >the data pulled from the log file, strung together as a `dict` and keyed by the _keys_

<a id="McUtils.McUtils.GaussianInterface.GaussianImporter.GaussianLogReader.get_default_keys" class="docs-object-method">&nbsp;</a>
```python
get_default_keys(self): 
```
Tries to get the default keys one might be expected to want depending on the type of job as determined from the Header
        Currently only supports 'opt', 'scan', and 'popt' as job types.
- `:returns`: `tuple(str)`
    >key listing

### Examples


