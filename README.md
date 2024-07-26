# McUtils [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mccoygroup/binder-mcutils/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fmccoygroup%252FMcUtils%26urlpath%3Dlab%252Ftree%252FMcUtils%252Fbinder%252Findex.ipynb%26branch%3Dmaster)

McUtils is a set of utilities written by the McCoy group for the McCoy group to handle common things we do, like pulling data from electronic structure calculations, doing unit conversions, interpolating functions, making attractive plots, getting finite difference derivatives, performing fast, vectorized operations, etc.

*This version is modified by Taras to include the SelectAnharmonicModes option*
#### TODO:
- make the FChkDerivatives class figure out the number of atoms smarter (from
  force constants and not from the total num of derivs)
- implement the smart insertion of zeroes into derivs
- implement smart insertion of zeroes into dipoles
- implement the selectanharmonicmodes option on the global level

We're working on [documenting the package](https://mccoygroup.github.io/References/Documentation/McUtils.html), but writing good documentation takes more time than writing good code.
Docs for the actively edited, unstable branch can be found [here](https://mccoygroup.github.io/McUtils).

### Installation & Requirements

The easiest way to install is via `pip`, as

```lang-shell
pip install mccoygroup-mcutils
```

This should install all dependencies. 
The major requirement is that Python 3.8+ is required due to use of features in the `types` module.
For safety, it is best to install this in a [virtual environment](https://docs.python.org/3.8/tutorial/venv.html), which we can make like

```lang-shell
python3.8 -m pip venv mcenv
```

and activate like

```lang-shell
. mcenv/bin/activate
```

or to use it in a [container](https://www.docker.com/) or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or some other place where we can control the environment.

It is also possible to install from source like

```lang-shell
git clone https://github.com/McCoyGroup/McUtils.git
```

but in this case you will need to make sure the library is on the path yourself and all of the dependencies are installed.
If you want to get all of the nice `JHTML` features for working in Jupyter, you'll then need to run

```python
from McUtils.Jupyter import JHTML
JHTML.load()
```

and then reload the browser window when prompted.

### Contributing

If you'd like to help out with this, we'd love contributions.
The easiest way to get started with it is to try it out.
When you find bugs, please [report them](https://github.com/McCoyGroup/McUtils/issues/new?title=Bug%20Found:&labels=bug). 
If there are things you'd like added [let us know](https://github.com/McCoyGroup/McUtils/issues/new?title=Feature%20Request:&labels=enhancement), and we'll try to help you get the context you need to add them yourself.
One of the biggest places where people can help out, though, is in improving the quality of the documentation.
As you try things out, add them as examples, either to the [main page](https://mccoygroup.github.io/References/Documentation/McUtils.html#examples) or to a [child page](https://mccoygroup.github.io/References/Documentation/McUtils/Plots/Plots/Plot.html#examples).
You can also edit the docstrings in the code to add context, explanation, argument types, return types, etc.
