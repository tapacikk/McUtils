from .LazyTensors import Tensor

__all__ = [
    'FunctionExpansion'
]

########################################################################################################################
#
#                                           FunctionExpansion
#
class FunctionExpansionException(Exception):
    pass
class FunctionExpansion:
    """A class for handling expansions of an internal coordinate potential up to 4th order
    Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian"""

    def __init__(self, derivatives, transforms = None):
        self._derivs = self.FunctionDerivatives(derivatives) # derivatives of the potential with respect to the cartesians
        if transforms is None:
            self._transf = None
        else:
             # transformation matrices from cartesians to internals
            self._transf = self.CoordinateTransforms(transforms)

    @property
    def tensors(self):
        return [ self.get_tensor(i) for i in range(len(self._derivs)) ]

    def expand(self, coords):
        """Returns a numerical value for the expanded coordinates

        :param coords:
        :type coords:
        :return:
        :rtype:
        """

        from functools import reduce

        c = Tensor(coords)
        return [ reduce(lambda cc, blah: t.dot(cc, axis = 1), range(i)) for i, t in enumerate(self.tensors) ]

    def __call__(self, coords):
        return sum(self.expand(coords))

    def get_tensor(self, i):
        # simple mat-vec
        if self._transf is None:
            term = self._derivs[i]
        elif i == 0:
            d0=self._derivs[0]
            term = self._transf[0]*d0 if d0 is not 0 else 0
        elif i == 1:
            terms = []
            d0 = self._derivs[0]
            d1 = self._derivs[1]
            if d0 is not 0: # we can be a bit clever about non-existant derivatives...
                terms.append(self._transf[1]*d0)
            if d1 is not 0:
                terms.append((self._transf[0] ** 2) * d1)

            term = sum(terms) if len(terms) > 0 else 0

        elif i == 2:

            terms = []
            d0 = self._derivs[0]
            d1 = self._derivs[1]
            d2 = self._derivs[2]
            if d0 is not 0: # we can be a bit clever about non-existant derivatives...
                terms.append(self._transf[2] * d0)
            if d1 is not 0:
                terms.append(
                    2 * ( self._transf[1] * self._transf[0] * d1 ) +
                        self._transf[0] * self._transf[1] * d1
                )
            if d2 is not 0:
                terms.append((self._transf[0] ** 3) * d2)

            term = sum(terms) if len(terms) > 0 else 0

        elif i == 3:

            terms = []
            d0 = self._derivs[0]
            d1 = self._derivs[1]
            d2 = self._derivs[2]
            d3 = self._derivs[3]

            if d0 is not 0: # we can be a bit clever about non-existant derivatives...
                terms.append(self._transf[3]*d0)
            if d1 is not 0:
                terms.append(
                    3*self._transf[2]*self._transf[0]*d1+
                    3*(self._transf[1]**2)*d1+
                    self._transf[0]*self._transf[2]*d1
                )
            if d2 is not 0:
                terms.append(
                    2*self._transf[0]*self._transf[1]*self._transf[0]*d2+
                    3*self._transf[1]*(self._transf[0]**2)*d2+
                    (self._transf[0]**2)*self._transf[1]*d2
                )
            if d3 is not 0:
                terms.append((self._transf[0]**4)*d3)

            term = sum(terms) if len(terms) > 0 else 0

        else:
            raise FunctionExpansionException("{}: expansion only provided up to order {}".format(type(self).__name__, 4))

        return term


    class CoordinateTransforms:
        def __init__(self, transforms):
            self._transf = [Tensor.from_array(t) for t in transforms]
        def __getitem__(self, i):
            if len(self._transf) < i:
                raise FunctionExpansionException("{}: transformations requested up to order {} but only provided to order {}".format(
                    type(self).__name__,
                    i,
                    len(self._transf)
                ))
            return self._transf[i]
        def __len__(self):
            return len(self._transf)
    class FunctionDerivatives:
        def __init__(self, derivs):
            self.derivs = [Tensor.from_array(t) for t in derivs]
        def __getitem__(self, i):
            if len(self.derivs) < i:
                raise FunctionExpansionException("{}: derivatives requested up to order {} but only provided to order {}".format(
                    type(self).__name__,
                    i,
                    len(self.derivs)
                ))
            return self.derivs[i]
        def __len__(self):
            return len(self.derivs)