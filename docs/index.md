# McUtils

McUtils is a set of utilities written by the McCoy group for the McCoy group to handle common things we do, like pulling data from electronic structure calculations, doing unit conversions, interpolating functions, making attractive plots, getting finite difference derivatives, performing fast, vectorized operations, etc.

We're working on [documenting the package](https://mccoygroup.github.io/References/Documentation/McUtils.html), but writing good documentation takes more time than writing good code.
Docs for the actively edited, unstable branch can be found [here](https://mccoygroup.github.io/McUtils).

### Installation & Requirements

McUtils is written in pure python. A small amount of C++ code has been added in the past, but we are trying to walk back from that, as it decreases ease of installation/portability.
We make use of `numpy`, `scipy`, and `matplotlib`, but have worked to avoid any dependencies beyond those three.

It is unlikely that McUtils will even find its way onto PyPI, so the best thing to do is install from GitHub via `git clone`. The `master` branch _should_ be stable. Other branches are intended to be development branches. 

### Contributing

If you'd like to help out with this, we'd love contributions.
The easiest way to get started with it is to try it out.
When you find bugs, please [report them](https://github.com/McCoyGroup/McUtils/issues/new?title=Bug%20Found:&labels=bug). 
If there are things you'd like added [let us know](https://github.com/McCoyGroup/McUtils/issues/new?title=Feature%20Request:&labels=enhancement), and we'll try to help you get the context you need to add them yourself.
One of the biggest places where people can help out, though, is in improving the quality of the documentation.
As you try things out, add them as examples, either to the [main page](https://mccoygroup.github.io/References/Documentation/McUtils.html#examples) or to a [child page](https://mccoygroup.github.io/References/Documentation/McUtils/Plots/Plots/Plot.html#examples).
You can also edit the docstrings in the code to add context, explanation, argument types, return types, etc.

  - [McUtils](McUtils.md)