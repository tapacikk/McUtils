"""Provides a general, convenient FiniteDifferenceFunction class to handle all of our difference FD imps

"""
import numpy as np

##########################################################################################
#
#                                    FiniteDifferenceFunction
#
##########################################################################################
class FiniteDifferenceError(Exception):
    pass
class FiniteDifferenceFunction:
    """The FiniteDifferenceFunction encapsulates a bunch of functionality extracted from Fornberger's
    Calculation of Wieghts in Finite Difference Formulas (https://epubs.siam.org/doi/pdf/10.1137/S0036144596322507)

    """
    REGULAR_GRID = "regular"
    IRREGULAR_GRID = "irregular"
    def __init__(self, coefficients, widths = None, order = None, gridtype = None, regularize_results = False,
                 mesh_spacings = None, num_vals = None
                 ):

        self._coefficients = coefficients # used to apply the weights to function values
        # NOTE: the coefficients should be supplied without 'h' if an evenly spaced grid
        # These will be forced back in later
        self._widths = self._clean_widths(widths) # used to determine how to apply the coefficients
        self._order = self._clean_order(order) # the order of the derivative
        if gridtype is None:
            gridtype = self.REGULAR_GRID
        self._gridtype = gridtype # if 'regular' will dispatch to 'call_evenly_spaced' otherwise we'll use more general code
        self._function = None
        self._regr = regularize_results
        self.mesh_spacings = mesh_spacings # in general we don't know this...
        self.num_vals = num_vals # in general we also don't know this...
    @property
    def gridtype(self):
        return self._gridtype
    @property
    def widths(self):
        return self._widths
    @property
    def order(self):
        return self._order
    @property
    def function(self):
        if self._function is None:
            self._function = self.get_FDF()
        return self._function
    @classmethod
    def RegularGridFunction(cls, n, accuracy = 4, end_point_precision = 4, dimension = None, **kw):
        """Constructs a finite-difference function that computes the nth derivative to a given accuracy order

        :param deriv:
        :type deriv:
        :param accuracy:
        :type accuracy:
        :return:
        :rtype:
        """

        # Gotta make sure we have the right dimensions for our derivatives
        if isinstance(n, int):
            if not isinstance(dimension, int):
                dimension = 1
            n = [ n ] * dimension
        else:
            dimension = len(n)
        if isinstance(accuracy, int):
            accuracy = [ accuracy ] * dimension

        coeff_list = [ None ] * dimension
        width_list = [ None ] * dimension
        for i, m in enumerate(n):
            stencil = m + accuracy[i]
            outer_stencil = stencil + end_point_precision
            lefthand_coeffs = cls._even_grid_coeffs(m, 0, outer_stencil)
            centered_coeffs = cls._even_grid_coeffs(m, np.math.ceil(stencil / 2), stencil)
            righthand_coeffs = cls._even_grid_coeffs(m, outer_stencil, outer_stencil)
            stencil = len(centered_coeffs)
            widths = [
                [0, len(lefthand_coeffs)],
                [np.math.ceil(stencil/2), np.math.floor(stencil/2)],
                [len(righthand_coeffs), 0]
            ]
            coeffs = [lefthand_coeffs, centered_coeffs, righthand_coeffs]
            coeff_list[i] = coeffs
            width_list[i] = widths

        fdf = cls(coeff_list, widths = width_list, order = n, gridtype = cls.REGULAR_GRID,
                  regularize_results= True, **kw
                  )
        return fdf

    @classmethod
    def IrregularGridFunction(cls, n, grid, stencil = 10, **kw):
        """Constructs a finite-difference function that computes the nth derivative with a given width

        :param deriv:
        :type deriv:
        :param accuracy:
        :type accuracy:
        :return:
        :rtype:
        """
        if stencil > len(grid):
            stencil = len(grid)

        # currently this is just fundamentally broken...
        # I'll need to recast this in terms of my _fdm type structure
        # so that these mats can be cleanly fed into the multidimensional FD stuff

        # we're gonna do just the 1D case for now
        slices = cls._pull_f_vals(grid, 0, stencil)
        slices = np.concatenate( (
            np.tile(slices[0], np.math.floor(stencil/2)),
            slices,
            np.tile(slices[0], np.math.ceil(stencil/2)),
        ) )
        coeffs = [ cls._uneven_spaced_weights(n, z, x) for z, x in zip(grid, slices)]
        widths = [ stencil ]

        coeff_list = [ coeffs ]
        width_list = [ widths ]
        return cls(coeff_list, widths = width_list, order = n, gridtype = cls.IRREGULAR_GRID, **kw )

    @classmethod
    def from_grid(cls, grid, order, stencil = None, gridtype = REGULAR_GRID):
        """

        :param grid:
        :type grid: np.ndarray
        :param values:
        :type values: np.ndarray
        :param order: order of the derivative to compute
        :type order: int or list of ints
        :param stencil: number of points to use in the stencil
        :type stencil: int or list of ints
        :param mode: 'regular' or 'irregular' specifying grid type
        :type mode: str
        :return:
        :rtype:
        """
        gp = np.asarray(grid)

        dim = max(len(grid.shape) - 1, 1)
        if dim > 1 and isinstance(order, int):
            order = (order,) * dim
        if stencil is None:
            if dim > 1:
                stencil = [o+2 for o in order]
            else:
                stencil = order + 2

        num_vals = gp.shape
        if len(gp.shape) > 1:
            num_vals = gp.shape[:-1]

        mode = gridtype.lower()
        # someday this should autodetect, but I don't really know how to do that...
        if mode != cls.IRREGULAR_GRID:
            ndd = gp.ndim
            if ndd > 1:
                ndd -= 1
                indsss = np.array([
                    [
                        [0]*(ndd) + [i],
                        [0]*(ndd-i-1) + [1] + [0]*(i) + [i]
                    ] for i in range(ndd)
                ])
                indsss = indsss.reshape(
                    (indsss.shape[0]*indsss.shape[1], )+indsss.shape[2:]
                ).T
                mesh_spacings = np.diff(gp[tuple(indsss)])[::2]
            else:
                mesh_spacings = np.array([
                  gp[1] - gp[0]
                ])
            if dim > 1:
                stencil = [n - o for n, o in zip(stencil, order)]
            return cls.RegularGridFunction(order, stencil, mesh_spacings = mesh_spacings, num_vals = num_vals)
        else:
            return cls.IrregularGridFunction(order, grid, stencil, num_vals = num_vals)

    def _clean_coeffs(self, cfs):
        cf1 = cfs[0]
        if isinstance(cf1, (int, float)):
            cfs = [ [ cfs ] ]
        return cfs
    def _clean_widths(self, ws):

        if isinstance(ws, (int, float)):
            ws = ( (int(ws), int(ws)) )
        else:
            w2s = [ None ] * len(ws)
            for i,w in enumerate(ws):
                if isinstance(w, (int, float)):
                    w2s[i] = (int(w), int(w))
                elif isinstance(w[0], (int, float)):
                    w2s[i] = (int(w[0]), int(w[1]))
                else:
                    w2s[i] = self._clean_widths(w)
            ws = tuple(w2s)
        return ws
    def _clean_order(self, order):
        if not isinstance(order, int):
            try:
                isinst = all((isinstance(o, int) for o in order))
            except:
                isinst = False
            if not isinst:
                raise TypeError("{}: order {} is expected to be an int or list of ints",
                                type(self).__name__, order)
        else:
            order = [ order ]
        return order

    #region Finite Difference Function Implementations
    def get_FDF(self):
        if self._gridtype != self.IRREGULAR_GRID:
            return self._evenly_spaced_FDF()
        else:
            return self._unevenly_spaced_FDF()

    def _evenly_spaced_FDF(self):
        """Generates a closure that applies the calculated coefficients to the case of an evenly spaced grid
         in a hopefully efficient manner

        :param h:
        :type h:
        :param function_values:
        :type function_values:
        :return:
        :rtype:
        """

        coeffs = self._coefficients
        orders = self._order
        return self._even_grid_FDF(coeffs, orders, mesh_spacings = self.mesh_spacings, num_vals = self.num_vals)
    @classmethod
    def _even_grid_FDF(cls, coeffs, orders, mesh_spacings = None, num_vals = None):
        """For simplicity this will handle mixed derivatives simply by doing each dimension separately

        :param take_lists:
        :type take_lists:
        :param coeffs:
        :type coeffs:
        :return:
        :rtype:
        """
        # num_vals is low-key the wrong name for this... it's really the number of points in the target values thing...

        # if we know the number of points ahead of time we can build our mats ahead of time
        # otherwise we need to be general about it
        if not num_vals is None:
            ndim = len(coeffs)
            if len(coeffs) > 1 and isinstance(num_vals, int):
                num_vals = [num_vals]*ndim
            if mesh_spacings is None:
                mmm = [ None ]*ndim
            else:
                mmm = mesh_spacings
            mats = [cls._fd_matrix_1D(*c, n, h = h, o = o ) for c, n, h, o in zip(coeffs, num_vals, mmm, orders)]
            if len(mats) == 1:
                mats = mats[0]
        else:
            mats = None

        def FDF(f_vals, h = mesh_spacings, mats = mats):
            "Calculates a finite difference"
            if mats is None:
                if h is None:
                    raise FiniteDifferenceError("{} object has no 'mesh_spacings' bound so one needs to be passed".format(cls.__name__))
                try:
                    len(h)
                except IndexError:
                    h = [h]*len(coeffs)
                meshy = h
                num_vals = f_vals.shape
                new_mats = [cls._fd_matrix_1D(*c, n, h = h, o = o ) for c, n, h, o in zip(coeffs, num_vals, meshy, orders)]
                if len(new_mats) == 1:
                    new_mats = new_mats[0]
                vavoom = cls._apply_fdm(new_mats, f_vals)
            else:
                vavoom = cls._apply_fdm(mats, f_vals)
            return vavoom

        return FDF

    def _unevenly_spaced_FDF(self):
        """Generates a closure that applies the calculated coefficients to the case of an evenly spaced grid
         in a hopefully efficient manner

        :param h:
        :type h:
        :param function_values:
        :type function_values:
        :return:
        :rtype:
        """

        coeffs = self._coefficients
        widths = self._widths
        orders = self._order
        return self._uneven_grid_FDF(self._pull_f_vals, coeffs, widths, orders)
    @staticmethod
    def _uneven_grid_FDF(take_lists, coeffs, widths, orders, regularize = False):
        """For simplicity this will handle mixed derivatives simply by doing each dimension separately

        :param take_lists:
        :type take_lists:
        :param coeffs:
        :type coeffs:
        :param widths:
        :type widths:
        :return:
        :rtype:
        """

        def FDF_1D(f_vals, w, c):
            """Calculates a 1D finite difference"""
            vals = np.asarray(f_vals)
            slices = take_lists(vals, 0, w)
            np.concatenate( (
                np.tile(slices[0], np.math.floor(w/2)),
                slices,
                np.tile(slices[0], np.math.ceil(w/2)),
            ) )
            fdf_vals = np.sum(slices*c, axis=1)
            return fdf_vals

        if len(widths) == 1:
            def FDF(f_vals):
                "Calculates a 1D finite difference"
                w = widths[0]
                c = coeffs[0]
                return FDF_1D(f_vals, w, c)
        else:
            # we do our finite differencing along multiple different axes as needed
            def FDF(f_vals):
                """Calculates the multi-dimensional FD by FD-ing each dimension in turn"""
                red_vals = f_vals
                for c, w in zip(coeffs, widths):
                    red_vals = FDF_1D(red_vals, c, w)
                return red_vals

        return FDF

    #endregion

    #region Finite Difference Matrices
    @classmethod
    def _fd_matrix_1D(cls, c_left, c_center, c_right, npts, h = None, o = None):
        """Builds a 1D finite difference matrix for a set of boundary weights, central weights, and num of points
        Will look like:
            b1 b2 b3 ...
            w1 w2 w3 ...
            0  w1 w2 w3 ...
            0  0  w1 w2 w3 ...
                 ...
                 ...
                 ...
                    .... b3 b2 b1

        :param coeffs:
        :type coeffs:
        :param npts:
        :type npts:
        :return:
        :rtype:
        """
        npts = int(npts)
        lcc = len(c_center)
        bound_l = int(np.floor(lcc/2))
        bdm_l = cls._fdm_core(c_left, bound_l, npts)
        bdm_r = cls._fdm_core(c_right, bound_l, npts, side = "r")
        core = cls._fdm_core(c_center, npts - lcc + (lcc % 2), npts)
        # print(core)
        fdm = np.concatenate((bdm_l, core, bdm_r))
        # print(h)
        if h is not None:
            fdm = fdm/(h**o) #useful for even grid cases where we'll cache the FD matrices for reuse
        return fdm
    @classmethod
    def _fdm_core(cls, a, n1, n2, side = "l"):
        # pulled from https://stackoverflow.com/a/52464135/5720002
        # returns the inner blocks of the FD mat
        a = np.asarray(a)
        p = np.zeros(n2, dtype=a.dtype)
        b = np.concatenate((p,a,p))
        s = b.strides[0]
        strided = np.lib.stride_tricks.as_strided
        if side == "r":
            mmm = strided(b[len(a)+n1-1:], shape=(n1, n2), strides=(-s,s))
            # print(mmm)
        else:
            mmm = strided(b[n2:], shape=(n1, n2), strides=(-s,s))
        return mmm
    @staticmethod
    def _apply_fdm(mats, vals):
        dim = len(mats) if isinstance(mats, (tuple, list)) else 1 # we use a non-numpy type as our flag we got more than one
        val_dim = len(vals.shape)
        if dim == 1: # we only have one FD matrix
            if val_dim == 1:
                vals = np.dot(mats, vals)
            elif val_dim == 2:
                vals = np.matmul(mats, vals)
            else:
                vals = np.tensordot(mats, vals, axes=((1), (0)))
        elif dim == 2 and val_dim == 2:
            # this is an easy common case so we'll do it directly
            np.savetxt("/Users/Mark/Desktop/pre.dat", vals)
            vals = np.matmul(mats[1], np.matmul(mats[0], vals).T).T
            np.savetxt("/Users/Mark/Desktop/wtf.dat", vals)
        else:
            # we'll assume dimension one is the slowest changing index, dimension 2 is the next slowest, etc.
            for i, m in enumerate(mats):
                vals = np.tensordot(m, vals, axes=((1), (i)))

        return vals

    #endregion

    #region Grid Weights
    @staticmethod
    def _StirlingS1_mat(n):
        # simple recursive definition of the StirlingS1 function in Mathematica
        # implemented at the C level mostly just for fun
        from .ZachLib import StirlingS1 # loads from C extension

        return StirlingS1(n)
    @staticmethod
    def _Binomial_mat(n):
        # simple recursive Binomial coefficients up to r, computed all at once to vectorize later ops
        # wastes space, justified by assuming a small-ish value for n
        from .ZachLib import Binomial # loads from C extension

        return Binomial(n)
    @staticmethod
    def _GammaBinomial_list(s, n):
        # Generalized binomial gamma function
        g = np.math.gamma
        g1 = g(s+1)
        g2 = np.array([g(m+1)*g(s-m+1) for m in range(n)])
        g3 = g1/g2
        return g3
    @staticmethod
    def _Factorial_list(n):
        # I was hoping to do this in some built in way with numpy...but I guess it's not possible?
        # looks like by default things don't vectorize and just call math.factorial
        base = np.arange(n, dtype=np.int64)
        base[0] = 1
        for i in range(1, n):
            base[i] = base[i]*base[i-1]
        return base
    @classmethod
    def _even_grid_coeffs(cls, m, s, n):
        """Finds the series coefficients for x^s*ln(x)^m centered at x=1. Uses the method:

             Table[
               Sum[
                ((-1)^(r - k))*Binomial[r, k]*
                    Binomial[s, r - j] StirlingS1[j, m] (m!/j!),
                {r, k, n},
                {j, 0, r}
                ],
               {k, 0, n}
               ]
             ]

        which is shown by J.M. here: https://chat.stackexchange.com/transcript/message/49528234#49528234
        """

        n = n+1 # in J.M.'s algorithm we go from 0 to n in Mathematica -- which means we have n+1 elements
        stirlings = cls._StirlingS1_mat(n)[:, m]
        bins = cls._Binomial_mat(n)
        sTest = s - int(s)
        if sTest == 0:
            bges = bins[ int(s) ]
        else:
            bges = cls._GammaBinomial_list(s, n)
        bges = np.flip(bges)
        facs = cls._Factorial_list(n)
        fcos = facs[m]/facs # factorial coefficient (m!/j!)

        import sys
        coeffs = np.zeros(n)
        for k in range(n):
            # each of these bits here should go from
            # Binomial[s, r - j] * StirlingS1[j, m] *
            bs = bges
            ss = stirlings
            fs = fcos
            bits = np.zeros(n-k)
            for r in range(k+1, n+1):
              bits[r-k-1] = np.dot(bs[-r:], ss[:r]*fs[:r])

            # (-1)^(r - k))*Binomial[r, k]
            cs = (-1)**(np.arange(n-k)) * bins[k:n, k]
            # print(bits, file=sys.stderr)
            coeffs[k] = np.dot(cs, bits)

        return coeffs
    @staticmethod
    def _uneven_spaced_weights(m, z, x):
        """Extracts the grid weights for an unevenly spaced grid based off of the algorithm outlined by
        Fronberger in https://pdfs.semanticscholar.org/8bf5/912bde884f6bd4cfb4991ba3d077cace94c0.pdf

        :param m: highest derivative order
        :type m:
        :param z: center of the derivatives
        :type z:
        :param X: grid of points
        :type X:
        """
        from .ZachLib import UnevenFiniteDifferenceWeights # loads from C extension

        x = np.asarray(x)
        return UnevenFiniteDifferenceWeights(m, z, x).T
    #endregion


##########################################################################################
#
#                                    finite_difference
#
##########################################################################################

def finite_difference(grid, values, order, stencil = None, gridtype = FiniteDifferenceFunction.REGULAR_GRID):
    """

    :param grid:
    :type grid: np.ndarray
    :param values:
    :type values: np.ndarray
    :param order: order of the derivative to compute
    :type order: int or list of ints
    :param stencil: number of points to use in the stencil
    :type stencil: int or list of ints
    :param mode: 'regular' or 'irregular' specifying grid type
    :type mode: str
    :return:
    :rtype:
    """
    gv = np.asarray(values)
    func = FiniteDifferenceFunction.from_grid(grid, order, stencil=stencil, gridtype=gridtype)
    fdf = func.function
    if func.gridtype != FiniteDifferenceFunction.IRREGULAR_GRID:
        return fdf(gv)
    else:
        return fdf(gv)