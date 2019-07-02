"""
Provides a general, convenient FiniteDifferenceFunction class to handle all of our difference FD imps
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
                 mesh_spacings = None, shape = None, only_core = False, only_center = False, axis = 0
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
        self.shape = shape # in general we also don't know this...
        self.only_core = only_core
        self.only_center = only_center
        self.axis = axis
    @property
    def coefficients(self):
        return self._coefficients
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
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
    @classmethod
    def RegularGridFunction(cls, n, accuracy = 4, stencil = None, end_point_precision = 4, dimension = None, **kw):
        """Constructs a finite-difference function that computes the nth derivative to a given accuracy order

        :param n: the order derivatives to use
        :type n: int or list(int)
        :param accuracy: the target accuracy of the method
        :type accuracy: int | Iterable[int]
        :param stencil: the number of points to use in the stencil
        :type stencil: None | int | Iterable[int]
        :param end_point_precision: the number of stencil points added to the end points
        :type end_point_precision: int
        :param dimension: the dimension of the derivative to apply
        :type dimension: int | Iterable[int]
        :param kw: keywords for initializing the class
        :type kw:
        :return: fdf
        :rtype: FiniteDifferenceFunction
        """

        # Gotta make sure we have the right dimensions for our derivatives
        if isinstance(n, (int, np.integer)):
            if not isinstance(dimension, (int, np.integer)):
                dimension = 1
            n = [ n ] * dimension
        else:
            dimension = len(n)

        # we're gonna try to force our derivatives to either use a set number of points (stencil) or
        # to have a certain order in the derivatives (accuracy)
        if isinstance(accuracy, (int, np.integer)):
            accuracy = [ accuracy ] * dimension
        if isinstance(stencil, (int, np.integer)) or stencil is None:
            stencil = [ stencil ] * dimension

        coeff_list = [ None ] * dimension
        width_list = [ None ] * dimension
        for i, m in enumerate(n):
            if m > 0:
                if stencil[i] is None:
                    sten = m + accuracy[i] - 1
                else:
                    sten = stencil[i] - 1

                outer_stencil = sten + end_point_precision
                lefthand_coeffs = cls._even_grid_coeffs(m, 0, outer_stencil)
                centered_coeffs = cls._even_grid_coeffs(m, np.math.ceil(sten / 2), sten)
                righthand_coeffs = cls._even_grid_coeffs(m, outer_stencil, outer_stencil)
                sten = len(centered_coeffs)
                widths = [
                    [0, len(lefthand_coeffs)],
                    [np.math.ceil(sten/2), np.math.floor(sten/2)],
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

    @staticmethod
    def get_mesh_spacings(gp):
        """Computes the mesh spacings of gp

        :param gp:
        :type gp: np.ndarray
        :return: mesh_spacings
        :rtype: np.ndarray
        """
        ndd = gp.ndim
        if ndd > 1: # ugly but workable way to calculate mesh_spacings
            ndd -= 1
            indsss = np.array([
                [
                    [0]*(ndd) + [i],
                    [0]*(i) + [1] + [0]*(ndd-i-1) + [i]
                ] for i in range(ndd)
            ])
            indsss = indsss.reshape(
                (indsss.shape[0]*indsss.shape[1], )+indsss.shape[2:]
            ).T
            mesh_spacings = np.diff(gp[tuple(indsss)])[::2]
        else:
            mesh_spacings = np.array([gp[1] - gp[0]])

        return mesh_spacings

    @classmethod
    def from_grid(cls, grid, order,
                  accuracy = 2, stencil = None, end_point_precision = 2, gridtype = REGULAR_GRID,
                  axis = 0
                  ):
        """Constructs a FiniteDifferenceFunction from a grid and order

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
        if axis is not None and axis > 0:
            ax = np.min(np.array([axis]))
            gp = gp[(0, )*np.min(ax)]
            # print(gp.shape)

        dim = max(len(gp.shape) - 1, 1)
        if dim > 1 and isinstance(order, (int, np.integer)):
            order = (order,) * dim
        if stencil is None:
            if dim > 1:
                if isinstance(accuracy, (int, np.integer)):
                    accuracy = [ accuracy ] * dim
                stencil = [o + a for o, a in zip(order, accuracy)]
            else:
                stencil = order + accuracy

        shape = gp.shape
        if len(gp.shape) > 1:
            shape = gp.shape[:-1]

        mode = gridtype.lower()
        # someday this should autodetect, but I don't really know how to do that...
        if mode != cls.IRREGULAR_GRID:
            mesh_spacings = cls.get_mesh_spacings(gp)
            # if dim > 1:
                # stencil = [n - o for n, o in zip(stencil, order)]
            return cls.RegularGridFunction(order,
                                           stencil = stencil,
                                           end_point_precision = end_point_precision,
                                           mesh_spacings = mesh_spacings,
                                           shape = shape,
                                           axis = axis
                                           )
        else:
            return cls.IrregularGridFunction(order, grid, stencil, shape = shape)

    def _clean_coeffs(self, cfs):
        cf1 = cfs[0]
        if isinstance(cf1, (int, float, np.integer, np.float)):
            cfs = [ [ cfs ] ]
        return cfs
    def _clean_widths(self, ws):

        if isinstance(ws, (int, float, np.integer, np.float)):
            ws = ( (int(ws), int(ws)) )
        else:
            w2s = [ None ] * len(ws)
            for i, w in enumerate(ws):
                if isinstance(w, (int, float, np.integer, np.float)):
                    w2s[i] = (int(w), int(w))
                elif w is None:
                    pass
                elif isinstance(w[0], (int, float, np.integer, np.float)):
                    w2s[i] = (int(w[0]), int(w[1]))
                elif w is not None:
                    w2s[i] = self._clean_widths(w)
            ws = tuple(w2s)
        return ws
    def _clean_order(self, order):
        if not isinstance(order, (int, np.integer)):
            try:
                isinst = all((isinstance(o, (int, np.integer)) for o in order))
            except:
                isinst = False
            if not isinst:
                raise TypeError("{}: order {} is expected to be an int or list of ints",
                                type(self).__name__, order)
        else:
            order = [ order ]
        return order

    #region Finite Difference Function Implementations
    def get_FDF(self, *args, **kwargs):
        if self._gridtype != self.IRREGULAR_GRID:
            return self._evenly_spaced_FDF(*args, **kwargs)
        else:
            return self._unevenly_spaced_FDF(*args, **kwargs)

    def _evenly_spaced_FDF(self, h = None, shape = None, only_core = None, only_center = None, axis = None):
        """Generates a closure that applies the calculated coefficients to the case of an evenly spaced grid
         in a hopefully efficient manner

         :param h:
         :type h:
         :param shape:
         :type shape:
         :param only_core:
         :type only_core:
         :return:
         :rtype:
        """

        coeffs = self._coefficients
        orders = self._order
        if h is None:
            h = self.mesh_spacings
        if only_core is None:
            only_core = self.only_core
        if only_center is None:
            only_center = self.only_center
        if axis is None:
            axis = self.axis
        if shape is None:
            shape = self.shape
        return self._even_grid_FDF(coeffs, orders, mesh_spacings = h, shape = shape,
                                   only_core = only_core, only_center = only_center, axis = axis)

    @classmethod
    def _even_grid_matrices(cls, shape, h, coeffs, orders, axis = None, only_core = False, only_center = False):
        if h is None:
            raise FiniteDifferenceError("{} object has no 'mesh_spacings' bound so one needs to be passed".format(cls.__name__))
        try:
            len(h)
        except (IndexError, TypeError):
            h = [h]*len(coeffs)
        meshy = h
        if axis is None:
            axis = 0

        num_vals = shape[axis:]

        new_mats = [
            cls._fd_matrix_1D(
                *c,
                n,
                h = h,
                o = o,
                only_core = only_core,
                only_center = only_center
                ) if (c is not None and o is not None) else None for c, n, h, o in zip(coeffs, num_vals, meshy, orders)
        ]
        # if len(new_mats) == 1:
        #     new_mats = new_mats[0]
        return new_mats

    @classmethod
    def _even_grid_FDF(cls, coeffs, orders, mesh_spacings = None, shape = None, only_core = None, only_center = None, axis = None):
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
        if shape is not None:
            ndim = len(coeffs)
            if axis is None:
                axis = 0
            if len(coeffs) > 1 and isinstance(shape, (int, np.integer)):
                shape = [shape]*ndim
            if len(shape) < axis + ndim: # padding for the call into _even_grid_matrices
                shape = (0,) *(ndim + axis - len(shape)) + tuple(shape)

            if mesh_spacings is None:
                mmm = [ None ]*ndim
            else:
                mmm = mesh_spacings
            mats = cls._even_grid_matrices(shape, mmm, coeffs, orders, axis = axis, only_core = only_core, only_center = only_center)
        else:
            mats = None

        def FDF(f_vals, h = None, mats = None, default_mats = mats, default_h = mesh_spacings, axis = axis,
                only_core = only_core,
                only_center = only_center
                ):
            "Calculates a finite difference"
            if mats is None:
                mats = default_mats
            if mats is None:
                if h is None:
                    h = default_h
                new_mats = cls._even_grid_matrices(f_vals.shape, h, coeffs, orders, axis = axis,
                                                   only_core = only_core, only_center = only_center
                                                   )
                vavoom = cls._apply_fdm(new_mats, f_vals, axis = axis)
            else:
                if h is not None and default_h is None:
                    if isinstance(h, (int, np.integer, float, np.float)):
                        h = [ h ] * len(mats)

                    mats = [ m / ( ms ** o) if m is not None else m for ms, m, o in zip(h, mats, orders) ]

                vavoom = cls._apply_fdm(mats, f_vals, axis = axis)
            return vavoom

        return FDF

    def _unevenly_spaced_FDF(self, *args, **kwargs):
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
    def _fd_matrix_1D(cls, c_left, c_center, c_right, npts, h = None, o = None, only_core = False, only_center = False):
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
        if only_core:
            fdm = cls._fdm_core(c_center, npts - lcc + (lcc % 2), npts)
        else:
            bound_l = min(int(np.floor(lcc/2)), int(np.floor(npts/2)))
            #TODO: sometimes the grid-point stencils overrun the grid I actually have (for small grids)
            # which means I need to handle these boundary cases better
            bdm_l = cls._fdm_core(c_left, bound_l, npts)
            bdm_r = cls._fdm_core(c_right, bound_l, npts, side = "r")
            core = cls._fdm_core(c_center, npts - lcc + (lcc % 2), npts)
            fdm = np.concatenate((bdm_l, core, bdm_r))

        if only_center:
            fdm_mid = int(np.floor(fdm.shape[0]/2))
            fdm = fdm[(fdm_mid, ), :]

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
    def _apply_fdm(mats, vals, axis = 0):

        dim = 1 if isinstance(mats, np.ndarray) else len(mats) # we use a non-numpy type as our flag we got more than one
        val_dim = len(vals.shape)

        if dim == 1 and mats is not None: # we only have one FD matrix

            if not isinstance(axis, (int, np.integer)):
                axis = axis[0]
            if not isinstance(mats, np.ndarray):
                mats = mats[0]
            if val_dim == 1:
                vals = np.dot(mats, vals) # axis doesn't even make sense here...
            else:
                vals = np.tensordot(mats, vals, axes=((1), (axis)))
                roll = np.concatenate((
                    np.roll(np.arange(axis+1), -1),
                    np.arange(axis+1, val_dim)
                ))
                vals = vals.transpose(roll)

        else:
            # by default we'll assume dimension one is the slowest changing index, dimension 2 is the next slowest, etc.
            if isinstance(axis, (int, np.integer)):
                axis = axis + np.arange(len(mats))

            for i, m in zip(axis, mats):
                if m is not None:
                    vals = np.tensordot(m, vals, axes=((1, ), (i, )))
                    roll = np.concatenate((
                        np.roll(np.arange(i+1), -1),
                        np.arange(i+1, val_dim)
                    ))
                    vals = vals.transpose(roll)
        return vals

    #endregion

    #region Grid Weights

    @staticmethod
    def _even_grid_coeffs(m, s, n):
        """Extracts the weights for an evenly spaced grid

        :param m:
        :type m:
        :param s:
        :type s:
        :param n:
        :type n:
        :return:
        :rtype:
        """
        from .ZachLib import EvenFiniteDifferenceWeights

        return EvenFiniteDifferenceWeights(m, s, n)
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

def finite_difference(grid, values, order,
                      accuracy = 4,
                      stencil = None,
                      end_point_precision = 4,
                      axis = None,
                      gridtype = FiniteDifferenceFunction.REGULAR_GRID,
                      **kw
                      ):
    """Computes a finite difference derivative for the values on the grid


    :param grid: the grid of points for which the vlaues lie on
    :type grid: np.ndarray
    :param values: the values on the grid
    :type values: np.ndarray
    :param order: order of the derivative to compute
    :type order: int | list[int]
    :param stencil: number of points to use in the stencil
    :type stencil: int | list[int]
    :param accuracy:
    :type accuracy:
    :param end_point_precision:
    :type end_point_precision:
    :param gridtype: 'regular' or 'irregular' specifying grid type
    :type gridtype: str
    :return:
    :rtype:
    """
    gv = np.asarray(values)
    func = FiniteDifferenceFunction.from_grid(
        grid, order,
        accuracy = accuracy,
        stencil = stencil,
        end_point_precision = end_point_precision,
        axis = axis,
        gridtype=gridtype,
        **kw
    )
    fdf = func.function
    return fdf(gv)
    # if func.gridtype != FiniteDifferenceFunction.IRREGULAR_GRID:
    #     return fdf(gv)
    # else:
    #     return fdf(gv)