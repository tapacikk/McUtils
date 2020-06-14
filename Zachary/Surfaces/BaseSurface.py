"""
Provides an abstract base class off of which concrete surface implementations can be built
"""

import abc, numpy as np

__all__ = [
    "BaseSurface",
    "TaylorSeriesSurface"
]

class BaseSurface(metaclass=abc.ABCMeta):
    """
    Surface base class which can be subclassed for relevant cases
    """
    def __init__(self, data, dimension):
        # we'll just give the core data a consistent attribute name
        self.data = data
        self.dimension = dimension
        # there's no 1000% general way to assess the dimension, so we just let it be passed
    @abc.abstractmethod
    def evaluate(self, points, **kwargs):
        """
        Evaluates the function at the points based off of "data"

        :param points:
        :type points:
        :return:
        :rtype:
        """
        raise NotImplemented
    def __call__(self, gridpoints, **kwargs):
        """

        :param gridpoints:
        :type gridpoints: np.ndarray
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.dimension is not None:
            gp_dim = gridpoints.shape[-1]
            if gp_dim != self.dimension:
                raise ValueError("{}: dimension mismatch in call, grid points had dim {} but surface expects dim {}".format(
                    type(self).__name__,
                    gp_dim,
                    self.dimension
                ))
        return self.evaluate(gridpoints, **kwargs)


class TaylorSeriesSurface(BaseSurface):
    """
    A surface with an evaluator built off of a Taylor series expansion
    """
    def __init__(self, derivs, opts=None, *, dimension=None):
        """
        :param data: derivs or a tuple of derivs + options
        :type data:
        :param dimension:
        :type dimension:
        """
        from ..Taylor import FunctionExpansion

        if opts is None:
            opts = {}
        if dimension is None:
            dimension = len(derivs[0])
        self.expansion = FunctionExpansion(derivs, **opts)
        opts["derivs"] = derivs
        super().__init__(opts, dimension)

    def evaluate(self, points, **kwargs):
        """
        Since the Taylor expansion stuff is already built out this is super easy

        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.expansion(points, **kwargs)

class LinearExpansionSurface(BaseSurface):
    """
    A surface with an evaluator built off of an expansion in some user specified basis
    """
    def __init__(self, coefficients, basis=None, dimension=None):
        """
        :param coefficients: the expansion coefficients in the basis
        :type coefficients: np.ndarray
        :param basis: a basis of functions to use (defaults to power series)
        :type basis: Iterable[function] | None
        """
        if basis is None:
            basis = [(lambda x,n=n: x**n) for n in range(len(coefficients))]
        self.basis = basis
        self.coeffs = coefficients
        super().__init__({"basis":basis, "coeffs":coefficients}, dimension)

    def evaluate(self, points, **kwargs):
        """
        First we just apply the basis to the gridpoints, then we dot this into the coeffs

        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        basis_values = [f(points, **kwargs) for f in self.basis]
        return np.dot(self.coeffs, basis_values)

class LinearFitSurface(LinearExpansionSurface):
    """
    A surface built off of a LinearExpansionSurface, but done by fitting.
    The basis selection
    """
    def __init__(self,
                 points,
                 basis=None,
                 order=4,
                 dimension=None
                 ):
        """
        :param points: a set of points to fit to
        :type points: np.ndarray
        :param basis: a basis of functions to use (defaults to power series)
        :type basis: Iterable[function] | None
        """
        from ..FittableModel import LinearFittableModel, LinearFitBasis
        dim = points.shape[-1]
        if basis is None:
            basis = LinearFitBasis([LinearFitBasis.power_series]*dim, order=order)
        self.model = LinearFittableModel(basis)
        self.model.fit(points)
        super().__init__({"basis":basis, "coeffs":coefficients}, dimension)

    def evaluate(self, points, **kwargs):
        """

        :param points:
        :type points:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        return self.model.evaluate(points)



