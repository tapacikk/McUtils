"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""

import itertools, typing, dataclasses
import sys

import numpy as np, abc, enum, math
import scipy.interpolate as interpolate
import scipy.optimize
import scipy.spatial as spat
from .Mesh import Mesh, MeshType
from ..Numputils import vec_outer, unique as nput_unique
# from ..Combinatorics import StirlingS1
from .Symbolic import TensorExpression

__all__ = [
    "Interpolator",
    "Extrapolator",
    "RBFDInterpolator",
    "ProductGridInterpolator",
    "UnstructuredGridInterpolator"
]


class InterpolatorException(Exception):
    pass


######################################################################################################
##
##                                   Interpolator Class
##
######################################################################################################

class BasicInterpolator(metaclass=abc.ABCMeta):
    """
    Defines the abstract interface we'll need interpolator instances to satisfy so that we can use
    `Interpolator` as a calling layer
    """

    @abc.abstractmethod
    def __init__(self, grid, values, **opts):
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def __call__(self, points, **kwargs):
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def derivative(self, order):
        """
        Constructs the derivatives of the interpolator at the given order
        :param order:
        :type order:
        :return:
        :rtype: BasicInterpolator
        """
        raise NotImplementedError("abstract interface")

class ProductGridInterpolator(BasicInterpolator):
    """
    A set of interpolators that support interpolation
    on a regular (tensor product) grid
    """

    def __init__(self, grids, vals, caller=None, order=None, extrapolate=True):
        """

        :param grids:
        :type grids:
        :param points:
        :type points:
        :param caller:
        :type caller:
        :param order:
        :type order: int | Iterable[int]
        """

        if order is None:
            order = 3

        self.grids = grids
        self.vals = vals
        if caller is None:
            if isinstance(grids[0], (int, float, np.integer, np.floating)):
                grids = [grids]
            ndim = len(grids)
            if ndim == 1:
                opts = {}
                if order is not None:
                    opts["k"] = order

                caller = interpolate.PPoly.from_spline(interpolate.splrep(grids[0], vals, k=order),
                                                      extrapolate=extrapolate
                                                      )
            else:
                caller = self.construct_ndspline(grids, vals, order, extrapolate=extrapolate)
        self.caller = caller

    @classmethod
    def construct_ndspline(cls, grids, vals, order, extrapolate=True):
        """
        Builds a tensor product ndspline by constructing a product of 1D splines

        :param grids: grids for each dimension independently
        :type grids: Iterable[np.ndarray]
        :param vals:
        :type vals: np.ndarray
        :param order:
        :type order: int | Iterable[int]
        :return:
        :rtype: interpolate.NdPPoly
        """

        # inspired by `csaps` python package
        # idea is to build a spline approximation in
        # every direction (where the return values are multidimensional)
        ndim = len(grids)
        if isinstance(order, (int, np.integer)):
            order = [order] * ndim

        coeffs = vals
        x = [None]*ndim
        for i, (g, o) in enumerate(zip(grids, order)):
            og_shape = coeffs.shape
            coeffs = coeffs.reshape((len(g), -1)).T
            sub_coeffs = [np.empty(0)]*len(coeffs)
            for e,v in enumerate(coeffs):
                ppoly = interpolate.PPoly.from_spline(interpolate.splrep(g, v, k=o))
                x[i] = ppoly.x
                sub_coeffs[e] = ppoly.c
            coeffs = np.array(sub_coeffs)
            coeffs = coeffs.reshape(
                og_shape[1:]+
                    sub_coeffs[0].shape
            )
            tot_dim = ndim+i+1
            coeffs = np.moveaxis(coeffs, tot_dim-2, tot_dim-2-i)

        return interpolate.NdPPoly(coeffs, x, extrapolate=extrapolate)

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype: np.ndarray
        """
        return self.caller(*args, **kwargs)

    def derivative(self, order):
        """
        :param order:
        :type order:
        :return:
        :rtype: ProductGridInterpolator
        """
        # ndim = len(self.grids)
        # if ndim == 1:
        return type(self)(
            self.grids,
            self.vals,
            caller=self.caller.derivative(order)
        )
        # elif ndim == 2:
        #     def caller(coords, _og=self.caller, **kwargs):
        #         return _og(coords, dx=order[0], dy=order[1], **kwargs)
        #     return type(self)(
        #         self.grids,
        #         self.vals,
        #         caller=caller
        #     )
        # else:
        #     derivs = self.caller.derivative(order)
        #     raise NotImplementedError("woof")

class UnstructuredGridInterpolator(BasicInterpolator):
    """
    Defines an interpolator appropriate for totally unstructured grids by
    delegating to the scipy `RBF` interpolators
    """

    default_neighbors=25
    def __init__(self, grid, values, order=None, neighbors=None, extrapolate=True, **opts):
        """
        :param grid:
        :type grid: np.ndarray
        :param values:
        :type values:  np.ndarray
        :param order:
        :type order: int
        :param neighbors:
        :type neighbors: int
        :param extrapolate:
        :type extrapolate: bool
        :param opts:
        :type opts:
        """
        self.extrapolate=extrapolate
        self._hull = None
        self._grid = grid

        if neighbors is None:
            neighbors = np.min([self.default_neighbors, len(grid)])

        if order is not None:
            if isinstance(order, int):
                if order == 1:
                    order = "linear"
                elif order == 3:
                    order = "cubic"
                elif order == 5:
                    order = "quintic"
                else:
                    raise InterpolatorException("{} doesn't support interpolation order '{}'".format(
                        interpolate.RBFInterpolator,
                        order
                    ))
            self.caller = interpolate.RBFInterpolator(grid, values, kernel=order, neighbors=neighbors, **opts)
        else:
            self.caller = interpolate.RBFInterpolator(grid, values, neighbors=neighbors, **opts)

    def _member_q(self, points):
        """
        Checks if the points are in the interpolated convex hull
        in the case that we aren't extrpolating so we can return
        NaN for those points
        :param points:
        :type points:
        :return:
        :rtype:
        """
        if self._hull is None:
            self._hull = spat.ConvexHull(self._grid)
            self._hull = spat.Delaunay(self._hull)
        return self._hull.find_simplex(points) >= 0
    def __call__(self, points):
        if self.extrapolate:
            return self.caller(points)
        else:
            hull_points = self._member_q(points)
            res = np.full(len(points), np.nan)
            res[hull_points] = self.caller(points[hull_points])
            return res

    def derivative(self, order):
        """
        Constructs the derivatives of the interpolator at the given order
        :param order:
        :type order:
        :return:
        :rtype: UnstructuredGridInterpolator
        """
        raise NotImplementedError("derivatives not implemented for unstructured grids")

# class _PolyFitManager:
#     """
#     Provides ways to evaluate polynomials
#     fast and get matrices of terms to fit
#     """
#     def __init__(self, order, ndim):
#         self.order = order
#         self.ndim = ndim
#         self._expr = None
#     @property
#     def perms(self):
#         from ..Combinatorics import SymmetricGroupGenerator
#
#         if self._perms is None:
#             # move this off to cached classmethod
#             # since there are restricted numbers of (order, ndim) pairs
#             sn = SymmetricGroupGenerator(self.ndim)
#             self._perms = sn.get_terms(list(range(self.order+1)))
#         return self._perms
#     def eval_poly_mat(self, coords):
#         """
#
#         :param coords: shape=(npts, ndim)
#         :type coords: np.ndarray
#         :return:
#         :rtype:
#         """
#         return np.power(coords[:, np.newaxis, :], self.perms[np.newaxis])
#     def eval_poly_mat_deriv(self, which):
#         which, counts = np.unique(which, return_counts=True)
#         perm_inds = self.perms[:, which]
#         shift_perms = perm_inds - counts[np.newaxis, :] # new monom orders


class RBFDInterpolator:
    """
    Provides a flexible RBF interpolator that also allows
    for matching function derivatives
    """

    def __init__(self,
                 pts, values, *derivatives,
                 kernel:typing.Union[callable,dict]='thin_plate_spline',
                 kernel_options=None,
                 auxiliary_basis=None,
                 extra_degree=0,
                 clustering_radius=.001,
                 monomial_basis=True,
                 multicenter_monomials=True
                 ):

        pts = np.asanyarray(pts)
        values = np.asarray(values)
        derivatives = [np.asanyarray(d) for d in derivatives]
        if pts.ndim == 1:
            pts = pts[:, np.newaxis]
        # if clustering_radius is None:
        #     clustering_radius = .01**(1/pts.shape[-1])
        if clustering_radius is not None and clustering_radius >= 0:
            pts, values, derivatives = self.decluster_data(pts, values, derivatives, clustering_radius)
        self.grid = pts
        self.tree = spat.KDTree(self.grid)
        self._pos_cache = {}
        self.vals = values
        self.derivs = derivatives
        if isinstance(kernel, str):
            kernel = self.default_kernels[kernel]
        if not isinstance(kernel, dict):
            kernel = {'function':kernel}
        if 'derivatives' not in kernel:
            kernel['derivatives'] = None
        if 'zero_handler' not in kernel:
            kernel['zero_handler'] = None
        if kernel_options is not None:
            kernel['function'] = lambda r,fn=kernel['function'],opt=kernel_options:fn(r,**opt)
            if kernel['derivatives'] is not None:
                kernel['derivatives'] = lambda n, fn=kernel['derivatives'],opt=kernel_options: (lambda r,d=fn(n):d(r, **opt))
            if kernel['zero_handler'] is not None:
                kernel['zero_handler'] = lambda n,ndim,fn=kernel['zero_handler'],opt=kernel_options: (lambda r,d=fn(n, ndim):d(r, **opt))
        self.kernel = kernel
        if not isinstance(auxiliary_basis, dict):
            auxiliary_basis = {'function':auxiliary_basis}
        if 'derivatives' not in auxiliary_basis:
            auxiliary_basis['derivatives'] = None
        self.aux_poly = auxiliary_basis
        self.extra_degree = extra_degree
        self.monomial_basis = monomial_basis
        self.multicenter_monomials = multicenter_monomials

        self._expr_cache = {}

        self._globint = None

    @classmethod
    def decluster_data(cls, pts, vals, derivs, radius):
        rpts, _, _ = cls.RescalingData.renormalize_grid(pts)
        dmat = np.linalg.norm(
            rpts[:, np.newaxis, :] - rpts[np.newaxis, :, :],
            axis=2
        )
        rad_pos = np.where(dmat<radius)
        non_diag = rad_pos[0] < rad_pos[1]
        rad_pos = (rad_pos[0][non_diag], rad_pos[1][non_diag])
        if len(rad_pos[0]) == 0:
            return pts, vals, derivs
        # for p in rad_pos
        # sort_pos = np.lexsort((rad_pos[1], rad_pos[0]))
        # rad_pos = (rad_pos[0][sort_pos], rad_pos[1][sort_pos])
        kill_pos = set()
        for r,k in zip(*rad_pos):
            if r not in kill_pos:
                kill_pos.add(k)
        kill_pos = np.array(list(kill_pos))

        mask = np.full(len(pts), True)
        mask[kill_pos] = False
        pts = pts[mask]
        vals = vals[mask]
        derivs = [d[mask] for d in derivs]

        return pts, vals, derivs

    @dataclasses.dataclass
    class RescalingData:
        grid_shifts:'Iterable[float]'
        grid_scaling:'Iterable[float]'
        vals_shift:'float'
        vals_scaling:'float'
        __slots__ = [
            'grid_shifts', 'grid_scaling',
            'vals_shift', 'vals_scaling'
        ]

        @classmethod
        def initialize_subgrid_data(cls, pts, values, derivatives):
            grid, grid_shifts, grid_scaling = cls.renormalize_grid(pts)
            vals, val_shift, val_scaling = cls.renormalize_values(values)
            derivs = cls.renormalize_derivs(derivatives, val_scaling, grid_scaling)
            return grid, vals, derivs, cls(grid_shifts, grid_scaling, val_shift, val_scaling)

        @classmethod
        def renormalize_grid(cls, pts):
            dim_slices = np.transpose(pts)
            mins = [np.min(x) for x in dim_slices]
            maxs = [np.max(x) for x in dim_slices]
            scalings = [M - m for M, m in zip(maxs, mins)]
            rescaled_slices = [(x - m) / s for x, m, s in zip(dim_slices, mins, scalings)]
            return np.transpose(rescaled_slices), np.array(mins), np.array(scalings)
        @classmethod
        def renormalize_values(cls, values):
            min = np.min(values)
            scaling = np.max(values) - min
            return (values - min) / scaling, min, scaling
        @classmethod
        def renormalize_derivs(cls, derivs, vals_scaling, grid_scaling):
            ders = []
            for n, values in enumerate(derivs):
                values = values / vals_scaling
                ndim = len(grid_scaling)
                gs = grid_scaling
                for d in range(n):
                    gs = np.expand_dims(gs, 0)
                    grid_scaling = grid_scaling[..., np.newaxis] * gs
                values = values * grid_scaling
                inds = (slice(None, None, None),) + tuple(
                    np.array(list(itertools.combinations_with_replacement(range(ndim), r=n + 1))).T
                )
                ders.append(values[inds])
            return ders

        def apply_renormalization(self, pts):
            pts = pts.T
            return ((pts - self.grid_shifts[:, np.newaxis]) / self.grid_scaling[:, np.newaxis]).T
        def reverse_renormalization(self, pts):
            return pts * self.grid_scaling + self.grid_shifts

        def reverse_value_renormalization(self, vs):
            return vs * self.vals_scaling + self.vals_shift

        def reverse_derivative_renormalization(self, derivs, reshape=True):
            ders = []
            npts = None
            ndim = len(self.grid_scaling)
            for n, values in enumerate(derivs):

                # we assume we have npts x (ndim,)*n so we reshape
                if npts is None: # we start with 1D
                    npts = len(values) / ndim
                    npp = int(npts)
                    if npts != npp:
                        raise ValueError("wtf")
                    npts = npp

                k = n+1

                shp = (npts,) + (ndim,) * k
                if reshape:
                    if np.prod(shp) == len(values):
                        values = values.reshape(shp)
                    else:
                        a = np.zeros(shp)
                        base_inds = RBFDInterpolator.triu_inds(ndim, k)
                        inds = (slice(None, None, None),) + base_inds
                        a[inds] = values.reshape((npts, -1))
                        for pos in np.array(base_inds).T:
                            base = (slice(None, None, None),) + tuple(pos)
                            for p in itertools.permutations(pos):
                                ind = (slice(None, None, None),) + p
                                a[ind] = a[base]
                        values = a
                else:
                    values = values.reshape((npts, -1))

                values = values * self.vals_scaling
                grid_scaling = self.grid_scaling
                gs = grid_scaling
                for d in range(n): # for a small performance boost I can avoid doing this repeatedly
                    gs = np.expand_dims(gs, 0)
                    grid_scaling = grid_scaling[..., np.newaxis] * gs
                if reshape:
                    values = values / grid_scaling
                else:
                    inds = RBFDInterpolator.triu_inds(ndim, k)
                    values = values / grid_scaling[inds]
                # inds = (slice(None, None, None),) + tuple(
                #     np.array(list(itertools.combinations_with_replacement(range(ndim), r=n + 1))).T
                # )
                ders.append(values)
            return ders

    @staticmethod
    def triu_inds(ndim, rank):
        return tuple(
            np.array(list(itertools.combinations_with_replacement(range(ndim), r=rank))).T
        )
    @staticmethod
    def triu_num(ndim, rank):
        if ndim == 1:
            return 1
        else:
            return math.comb(rank+(ndim-1), (ndim-1))

    @staticmethod
    def gaussian(r, e=1):
        return np.exp(-(r**2)/(2*e**2))
    @staticmethod
    def gaussian_derivative(n:int):
        from scipy.special import hermite
        s = -1 if n%2 == 1 else 1
        def deriv(r, e=1):
            w = 1 / (e * np.sqrt(2))
            r = w * r
            return s*(w**n)*hermite(n)(r)*np.exp(-r**2)
        return deriv
    @staticmethod
    def gaussian_singularity_handler(n: int, ndim: int):
        # handler for what happens at zero specifically
        def deriv(r, e=1): # r will be a vector of zeros...
            dmat = np.zeros((len(r),) + (ndim,)*n)
            if n % 2 == 0:
                crds = np.arange(ndim)
                # take tuples of crds, could be faster but we shouldn't
                # need many evals of this...
                for tup in itertools.combinations_with_replacement(crds, r=n//2):
                    idx, counts = np.unique(tup, return_counts=True)
                    val = np.product([np.arange(1, 2*c+1, 2) for c in counts])/(e**n) # rising factorial
                    idx = sum(
                        [
                            [i] * (2 * c) for i, c in
                            zip(idx, counts)
                        ],
                        []
                    )
                    for p in itertools.permutations(idx):
                        p = (slice(None, None, None),) + p
                        dmat[p] = val
            return ((-1)**(n//2))*dmat
        return deriv

    @staticmethod
    def thin_plate_spline(r, o=3):
        return r**o * (1 if o%2 == 1 else np.log(r+1))
    @staticmethod
    def thin_plate_spline_derivative(n: int):
        def deriv(r, o=3):
            if o % 2 == 1:
                k = np.product(o-np.arange(n))
                return k * r**(o-n)
            elif n == 1: # clearly could work this out but I don't want to right now
                """r^n/(1 + r) + n r^(-1 + n) Log[1 + r]"""
                return r**n/(1+r) + n*r**(n-1)*np.log(1+r)
            elif n == 2:
                """-(r^n/(1 + r)^2) + (2 n r^(-1 + n))/(1 + r) + (-1 + n) n r^(-2 + n) Log[1 + r]"""
                return -(r**n/(1+r)**2) + (2*n*r**(n-1))/(1+r) + n*(n-1)*r**(n-2)*np.log(1+r)
            else:
                raise NotImplementedError("need higher derivs for even thin plate splines...")
        return deriv
    @staticmethod
    def thin_plate_spline_singularity_handler(n: int, ndim: int):
        # handler for what happens at zero specifically
        def deriv(r, o=3):  # r will be a vector of zeros...
            if n - (1 - (o%2)) > o:
                raise ValueError("thin plate spline deriv not defined")
            return  np.zeros((len(r),) + (ndim,) * n)
        return deriv
    @property
    def default_kernels(self):
        return {
            'gaussian': {
                'function': self.gaussian,
                'derivatives': self.gaussian_derivative,
                'zero_handler': self.gaussian_singularity_handler
            },

            'thin_plate_spline': {
                'function': self.thin_plate_spline,
                'derivatives': self.thin_plate_spline_derivative,
                'zero_handler': self.thin_plate_spline_singularity_handler
            }
        }

    def _poly_exprs(self, ndim:int, order:int, deriv_order:int, monomials=True):
        k = (ndim, order, deriv_order)
        if k not in self._expr_cache:
            if monomials:
                base_expr = TensorExpression.ScalarPowerTerm(TensorExpression.CoordinateVector(ndim, name='coord_vec'), order)
            else:
                base_expr = TensorExpression.OuterPowerTerm(TensorExpression.CoordinateVector(ndim, name='coord_vec'), order)
            exprs = [base_expr]
            for _ in range(deriv_order):
                exprs.append(exprs[-1].dQ())
            self._expr_cache[k] = exprs
        return self._expr_cache[k]
    def evaluate_poly_matrix(self, pts, degree, deriv_order=0, poly_origin=0.5, include_constant_term=True, monomials=True):
        if isinstance(poly_origin, (int, float, np.integer, np.floating)):
            pts = pts - poly_origin
        else:
            poly_origin = np.asanyarray(poly_origin)
            pts = pts[np.newaxis, :, :] - poly_origin[:, np.newaxis, :]

        ndim = pts.shape[-1]
        exprs = (
                (
                    [    # add in constant term by hand
                        [
                            TensorExpression.ConstantArray(np.ones(pts.shape[:-1] + (1,)))
                        ] +
                        [
                            TensorExpression.ConstantArray(np.zeros(pts.shape[:-1] + (ndim,)*d + (1,)) )
                            for d in range(1, deriv_order+1)
                        ]
                    ]
                    if include_constant_term else
                    []
                ) +
                [self._poly_exprs(ndim, d, deriv_order, monomials=monomials) for d in range(1, degree + 1)]
        )

        crds = TensorExpression.ArrayStack(pts.shape[:-1], pts)
        vals = [ # evaluate poly values & derivs at each order in a dumb, kinda inefficient way? (for now)
            [
                TensorExpression(e, coord_vec=crds).eval()
                for e in exp_list
            ]
            for exp_list in exprs
        ]

        # Set things up so that arbitrary numbers of centers can be used at once
        padding = (slice(None, None, None),) * (2 if isinstance(poly_origin, np.ndarray) else 1)

        # now we take only upper triangle indices to get only unique ones
        flat_vals = [[] for _ in range(deriv_order+1)]
        for o, v_list in enumerate(vals): # loop over different stacks
            # get upper inds for the OuterPower term
            if not include_constant_term or o > 0:
                if not include_constant_term:
                    o = o+1
                if monomials:
                    power_inds = ()
                else:
                    power_inds = self.triu_inds(ndim, o)
                pis = padding + power_inds
                flat_vals[0].append(v_list[0][pis])
            else:
                power_inds = ()
                flat_vals[0].append(v_list[0]) # need to add a dimension for joining...
            for n,d in enumerate(v_list[1:]): #need utri inds
                # sample first power inds then derivs
                pis = padding + (slice(None, None, None),)*(n+1) + power_inds
                d = d[pis]
                # get upper inds for the different derivatives
                inds = self.triu_inds(ndim, n+1)
                inds = padding + inds
                d = d[inds]
                flat_vals[n+1].append(d)

        mats = [np.concatenate(v, axis=-1) for v in flat_vals]
        if isinstance(poly_origin, np.ndarray):
            mats = [np.moveaxis(m, 0, -1) for m in mats]
            mats = [np.reshape(m, m.shape[:-2] + (-1,)) for m in mats]
        mat = np.concatenate(
            [
                m.reshape((-1, m.shape[-1]))
                for m in mats
            ],
            axis=0
        )
        return mat

    def _rbf_exprs(self, ndim, deriv_order):
        k = (ndim, deriv_order)
        if k not in self._expr_cache:
            exprs = [
                TensorExpression.ScalarFunctionTerm(
                    TensorExpression.VectorNormTerm(
                        TensorExpression.CoordinateVector(ndim, name="pts")
                    ),
                    f=self.kernel
                )
            ]
            for _ in range(deriv_order):
                exprs.append(exprs[-1].dQ())
            self._expr_cache[k] = exprs
        return self._expr_cache[k]
    def evaluate_rbf_matrix(self, pts, centers, deriv_order=0, zero_tol=1e-8):
        displacements = np.reshape(pts[:, np.newaxis, :] - centers[np.newaxis, :, :], (-1, centers.shape[-1]))
        distances = np.linalg.norm(displacements, axis=1)
        zero_pos = np.where(np.abs(distances) <= zero_tol) # we don't feed these displacements into the derivative evals
        nonz_pos = np.where(np.abs(distances) > zero_tol) # this could be fast but w/e
        displacements = displacements[nonz_pos]

        # now we evaluate the norm term...
        ndim = displacements.shape[-1]
        crds = TensorExpression.ArrayStack((len(displacements),), displacements)
        expr_vals = [
            TensorExpression(e, pts=crds).eval()
            for e in self._rbf_exprs(ndim, deriv_order)
        ]
        flat_expr_vals = [expr_vals[0]]
        for n, d in enumerate(expr_vals[1:]):  # need utri inds
            # get upper inds for the different derivatives
            inds = self.triu_inds(ndim, n+1)# tuple(np.array(list(itertools.combinations_with_replacement(range(ndim), r=n + 1))).T)
            inds = (slice(None, None, None),) + inds
            d = d[inds]
            flat_expr_vals.append(d)

        # handle singularities
        if len(zero_pos) > 0:
            zv = np.zeros(len(zero_pos[0]))
            # now we manage the singularities in the derivs
            sing_vals = [self.kernel['function'](zv)]
            handler = self.kernel['zero_handler']
            for n in range(1, deriv_order+1):
               d = handler(n, ndim)(zv)
               inds = self.triu_inds(ndim, n)#tuple(np.array(list(itertools.combinations_with_replacement(range(ndim), r=n))).T)
               inds = (slice(None, None, None),) + inds
               d = d[inds]
               sing_vals.append(d)

            #create final vectors by merge datasets appropriately
            full_exprs = []
            for v,s in zip(flat_expr_vals, sing_vals):
                a = np.zeros( (len(distances),) + v.shape[1:] )
                a[zero_pos] = s
                a[nonz_pos] = v
                full_exprs.append(a)
            flat_expr_vals = full_exprs

        # now reshape to turn into full mats
        npts = len(centers)

        flat_expr_vals = [
            np.moveaxis(
                a.reshape((len(pts), len(centers)) + a.shape[1:]),
                1, -1
            )
            for a in flat_expr_vals
        ]
        mat = np.concatenate(
            [
                a.reshape((-1, npts))
                for a in flat_expr_vals
            ],
            axis=0
        )

        return mat
    def construct_matrix(self,
                         pts,
                         centers,
                         degree=0, # passed independently
                         deriv_order=0, # passed independently
                         zero_tol=1e-8, # not really intended to be changed...
                         poly_origin=0.5, # can be set at the class level
                         include_constant_term=True, # can be set at the class level
                         force_square=False,
                         monomials=True,
                         multicentered_polys=False
                         ):
        if degree > 0:
            pol_mats = self.evaluate_poly_matrix(pts, degree,
                                                 poly_origin=poly_origin if not multicentered_polys else centers,
                                                 include_constant_term=include_constant_term,
                                                 deriv_order=deriv_order,
                                                 monomials=monomials)
        else:
            pol_mats = None
        rbf_mats = self.evaluate_rbf_matrix(pts, centers, deriv_order=deriv_order, zero_tol=zero_tol)
        if pol_mats is not None:
            rbf_mats = np.concatenate([rbf_mats, pol_mats], axis=1)
        if force_square and rbf_mats.shape[1] > rbf_mats.shape[0]: # force it to be square by including extra polys
            curl = rbf_mats.shape[0]
            neel = rbf_mats.shape[1]
            disp = neel-curl
            rbf_mats = np.concatenate([rbf_mats, np.zeros((disp, neel))], axis=0)
            rbf_mats[curl:, :curl] = rbf_mats[:curl, -disp:].T
        return rbf_mats

    def solve_system(self, centers, vals, derivs:list, solver='least_squares', return_data=False):
        if len(vals) != len(centers):
            raise ValueError("need same number of interpolation points and values (got {} and {})".format(
                len(centers),
                len(vals)
            ))
        # TODO: compute the appropriate deriv order to match the vals and derivs
        derivs = [d.flatten() for d in derivs]
        nder = len(derivs)
        vals = np.concatenate([vals]+derivs, axis=0)

        degree = 0
        if nder > 0:
            # I now know how many derivs I have per order...but I get the same contrib from each
            # degree order, but the issue is the num derivs scales with the number of centers too
            # so it needs to be solved for differently
            ndim = centers.shape[-1]
            target_extra = 1 + len(centers)*sum(self.triu_num(ndim, k) for k in range(1, nder))
            cur = 0
            ncent = len(centers) if self.multicenter_monomials else 1
            for degree in range(1, nder*(target_extra//ndim)):# not sure where I need to stop?
                cur = cur + (self.triu_num(ndim, degree) if not self.monomial_basis else ndim*ncent)
                if cur >= target_extra:
                    break

        M = self.construct_matrix(centers, centers,
                                  degree=degree+self.extra_degree,
                                  deriv_order=nder,
                                  monomials=self.monomial_basis,
                                  multicentered_polys=self.multicenter_monomials,
                                  force_square=True) # deriv_order and degree collaborate to get a square matrix...?
        if len(M) > len(vals):
            vals = np.concatenate([vals, np.zeros(len(M) - len(vals))])
        # with np.printoptions(linewidth=1e8):
        #     print(centers)
        #     print(M)
        #     print(vals[:, np.newaxis])
        if solver == "least_squares":
            w_data = np.linalg.lstsq(M, vals, rcond=None)
        else:
            w_data = np.linalg.solve(M, vals)#, rcond=None)
            w_data = (w_data,)

        if return_data:
            data = (M, )
            return w_data, degree+self.extra_degree, data
        else:
            return w_data, degree+self.extra_degree

    def get_neighborhood(self, pts, *, neighbors):
        _, yindices = self.tree.query(pts, k=neighbors)
        if neighbors == 1:
            # `KDTree` squeezes the output when neighbors=1.
            yindices = yindices[np.newaxis]
        yindices = np.sort(yindices)
        return yindices

    def create_neighbor_groups(self, inds, merge_limit=3):
        #TODO: lots of optimizations here, cutting down on loops, making use of sorting
        #      not doing useless setdiffs, etc.
        # _, sorting, subinds = nput_unique(inds, axis=0, return_index=True)
        inds, sorting, counts = nput_unique(inds, axis=0, return_counts=True)
        subinds = np.split(sorting, np.cumsum(counts))[:-1]
        # raise Exception(nput_unique(inds, axis=0, return_index=True))
        allowed_diffs = len(inds[0]) - merge_limit
        if allowed_diffs > 0:
            groups = []
            idx_mapping = []
            for i,idx in zip(subinds, inds):
                for n,g in enumerate(groups):
                    diff = np.setdiff1d(idx, g, assume_unique=True)
                    if len(diff) < allowed_diffs:
                        groups[n] = np.sort(np.concatenate([g, diff]))
                        idx_mapping[n] = np.concatenate([idx_mapping[n], i])
                        break
                else:
                    groups.append(idx)
                    idx_mapping.append(i)
        else:
            groups = inds
            idx_mapping = subinds
        # print(np.cumsum(counts),
        #       idx_mapping)

        return groups, idx_mapping

    class InterpolationData:
        __slots__ = ['weights', 'centers', 'degree', 'scaling_data', 'solver_data']
        def __init__(self, w, grid, degree, scaling_data, solver_data=None):
            self.weights = w
            self.centers = grid
            self.degree = degree
            self.scaling_data = scaling_data
            self.solver_data = solver_data
        # def eval(self, *args, **kwargs):
        #     grid = self.grid[vals_inds]
        #     vals = self.vals[vals_inds]
        #     derivs = [d[vals_inds] for d in self.derivs]
        #
        #     grid, vals, derivs, scaling_data = self.RescalingData.initialize_subgrid_data(
        #         grid,
        #         vals,
        #         derivs
        #     )
        #
        #     # print(vals_inds)
        #     # print(vals)
        #     # print(derivs)
        #
        #     w, degree = self.solve_system(grid, vals, derivs)
        #
        #     self._pos_cache[k] = (w, grid, degree, scaling_data)

    def construct_evaluation_matrix(self, pts, data, deriv_order=0):
        """
        :param pts:
        :type pts:
        :param data:
        :type data:
        :param deriv_order:
        :type deriv_order:
        :return:
        :rtype:
        """

        scaling_data = data.scaling_data
        # w = data.weights
        degree = data.degree
        centers = data.centers

        sub_pts = scaling_data.apply_renormalization(pts)
        M = self.construct_matrix(sub_pts, centers, degree=degree, deriv_order=deriv_order, monomials=self.monomial_basis, multicentered_polys=self.multicenter_monomials)
        return M

    def apply_interpolation(self, pts, data, reshape_derivatives=True, deriv_order=0):
        """

        :param pts:
        :type pts:
        :param data:
        :type data:
        :param deriv_order:
        :type deriv_order:
        :return:
        :rtype:
        """

        scaling_data = data.scaling_data
        w = data.weights
        degree = data.degree
        centers = data.centers

        sub_pts = scaling_data.apply_renormalization(pts)
        M = self.construct_matrix(sub_pts, centers, degree=degree,
                                  deriv_order=deriv_order,
                                  monomials=self.monomial_basis,
                                  multicentered_polys=self.multicenter_monomials
                                  )
        v = np.dot(M, w[0])

        npts = len(sub_pts)
        vals = scaling_data.reverse_value_renormalization(v[:npts])
        dvs = []
        if deriv_order > 0:  # derivatives
            offset = npts
            ndim = pts.shape[-1]
            for o in range(1, 1 + deriv_order):
                num = npts * self.triu_num(ndim, o)
                dvs.append(v[offset:offset + num])
                # print(dvs[-1])
                offset += num
            dvs = scaling_data.reverse_derivative_renormalization(dvs, reshape=reshape_derivatives)

        return vals, dvs

    def construct_interpolation(self, inds, solver_data=False):
        grid = self.grid[inds]
        vals = self.vals[inds]
        derivs = [d[inds] for d in self.derivs]

        grid, vals, derivs, scaling_data = self.RescalingData.initialize_subgrid_data(
            grid,
            vals,
            derivs
        )

        res = self.solve_system(grid, vals, derivs, return_data=solver_data)
        if solver_data:
            w, degree, data = res
        else:
            w, degree = res
            data = None

        new = self.InterpolationData(w, grid, degree, scaling_data, solver_data=data)
        return new

    class Interpolator:
        __slots__ = ['data', 'parent']
        def __init__(self, data:'RBFDInterpolator.InterpolationData', parent:'RBFDInterpolator'):
            self.data = data
            self.parent = parent
        def __call__(self, pts, deriv_order=0, reshape_derivatives=True):
            pts = np.asanyarray(pts)
            smol = pts.ndim == 1
            if smol:
                pts = pts[np.newaxis]
            vals, ders = self.parent.apply_interpolation(pts, self.data,
                                                   deriv_order=deriv_order,
                                                   reshape_derivatives=reshape_derivatives
                                                   )

            if deriv_order > 0:
                return [vals] + ders
            else:
                return vals

        def matrix(self, pts, deriv_order=0):
            pts = np.asanyarray(pts)
            smol = pts.ndim == 1
            if smol:
                pts = pts[np.newaxis]
            return self.parent.construct_evaluation_matrix(pts, self.data, deriv_order=deriv_order)

    def nearest_interpolation(self, pts, neighbors=15, solver_data=False, interpolator=True):
        """

        :param pts:
        :type pts:
        :param neighbors:
        :type neighbors:
        :param solver_data:
        :type solver_data:
        :param interpolator:
        :type interpolator:
        :return:
        :rtype: RBFDInterpolator.Interpolator|RBFDInterpolator.InterpolationData
        """
        pts = np.asanyarray(pts)
        smol = pts.ndim == 1
        if smol:
            pts = pts[np.newaxis]
        inds = self.get_neighborhood(pts, neighbors=neighbors)
        dats = [self.construct_interpolation(i, solver_data=solver_data) for i in inds]
        if interpolator:
            dats = [self.Interpolator(d, self) for d in dats]
        if smol:
            dats = dats[0]
        return dats

    def eval(self, pts, deriv_order=0, neighbors=15, merge_neighbors=5, reshape_derivatives=True):
        pts = np.asanyarray(pts)
        if pts.ndim == 1:
            pts = pts[np.newaxis]

        ind_grps, pts_grps = self.create_neighbor_groups(
            self.get_neighborhood(pts, neighbors=neighbors),
            merge_limit=merge_neighbors
        )
        # print(ind_grps, pts_grps)

        val_sets = np.empty(len(pts))
        der_sets = None
        for pts_inds, vals_inds in zip(pts_grps, ind_grps): # we need to evaluate this entire loop for each point......
            k = tuple(vals_inds)
            if k not in self._pos_cache:
                self._pos_cache[k] = self.construct_interpolation(vals_inds)
            vals, ders = self.apply_interpolation(pts[pts_inds], self._pos_cache[k], deriv_order=deriv_order, reshape_derivatives=reshape_derivatives)
            val_sets[pts_inds] = vals
            if der_sets is None:
                der_sets = [
                    np.empty((len(pts),) + d.shape[1:])
                    for d in ders
                ]
            for n, d in enumerate(ders):
                der_sets[n][pts_inds] = d

        if deriv_order > 0:
            return [val_sets] + der_sets
        else:
            return val_sets

    def __call__(self, pts, deriv_order=0, neighbors=15, merge_neighbors=5, reshape_derivatives=True):
        return self.eval(pts, deriv_order=deriv_order, neighbors=neighbors, merge_neighbors=merge_neighbors, reshape_derivatives=reshape_derivatives)

    @property
    def global_interpolator(self):
        if self._globint is None:
            self._globint = self.Interpolator(
                self.construct_interpolation(np.arange(len(self.grid))),
                self
            )
        return self._globint


    @classmethod
    def create_function_interpolation(cls,
                                      pts,
                                      fn,
                                      *derivatives,
                                      derivative_order=None,
                                      function_shape=None,
                                      **opts):
        from ..Scaffolding import ParameterManager
        from .Taylor import FiniteDifferenceDerivative

        if derivative_order is None:
            derivative_order = len(derivatives)
        vals = fn(pts)
        dvals = []
        for n,d in enumerate(derivatives):
            if n < derivative_order:
                dvals.append(d(pts))
            else:
                break

        if pts.ndim == 1:
            input_shape = 0
        else:
            input_shape = pts.shape[-1]

        opts = ParameterManager(opts)
        if len(dvals) < derivative_order:
            fd_opts = opts.filter(FiniteDifferenceDerivative)
            base_deriv_order = len(derivatives)
            fdd = FiniteDifferenceDerivative(
                derivatives[-1] if base_deriv_order > 0 else fn,
                function_shape=(input_shape, 0) if function_shape is None else function_shape
            )

            deriv_tensors = fdd.derivatives(
                center=pts,
                **fd_opts
            ).compute_derivatives(list(range(1, derivative_order - base_deriv_order + 1)))

            dvals.extend(
                np.moveaxis(a, 0, -1) for a in deriv_tensors[base_deriv_order:]
            )

        return cls(
            pts,
            vals,
            *dvals,
            **opts.exclude(FiniteDifferenceDerivative)
        )


class ExtrapolatorType(enum.Enum):
    Default='Automatic'
    Error='Raise'

class Interpolator:
    """
    A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work
    """
    DefaultExtrapolator = ExtrapolatorType.Default
    def __init__(self,
                 grid,
                 vals,
                 interpolation_function=None,
                 interpolation_order=None,
                 extrapolator=None,
                 extrapolation_order=None,
                 **interpolation_opts
                 ):
        """
        :param grid: an unstructured grid of points **or** a structured grid of points **or** a 1D array
        :type grid: np.ndarray
        :param vals: the values at the grid points
        :type vals: np.ndarray
        :param interpolation_function: the basic function to be used to handle the raw interpolation
        :type interpolation_function: None | BasicInterpolator
        :param interpolation_order: the order of extrapolation to use (when applicable)
        :type interpolation_order: int | str | None
        :param extrapolator: the extrapolator to use for data points not on the grid
        :type extrapolator: Extrapolator | None | str | function
        :param extrapolation_order: the order of extrapolation to use by default
        :type extrapolation_order: int | str | None
        :param interpolation_opts: the options to be fed into the interpolating_function
        :type interpolation_opts:
        """
        self.grid = grid = Mesh(grid) if not isinstance(grid, Mesh) else grid
        self.vals = vals
        if interpolation_function is None:
            interpolation_function = self.get_interpolator(grid, vals,
                                                           interpolation_order=interpolation_order,
                                                           allow_extrapolation=extrapolator is None,
                                                           **interpolation_opts
                                                           )
        self.interpolator = interpolation_function

        if extrapolator is not None:
            if isinstance(extrapolator, ExtrapolatorType):
                if extrapolator == ExtrapolatorType.Default:
                    extrapolator = self.get_extrapolator(grid, vals, extrapolation_order=extrapolation_order)
                else:
                    raise ValueError("don't know what do with extrapolator type {}".format(extrapolator))
            elif not isinstance(extrapolator, Extrapolator):
                extrapolator = Extrapolator(extrapolator)
        self.extrapolator = extrapolator

    @classmethod
    def get_interpolator(cls, grid, vals, interpolation_order=None, allow_extrapolation=True, **opts):
        """Returns a function that can be called on grid points to interpolate them

        :param grid:
        :type grid: Mesh
        :param vals:
        :type vals: np.ndarray
        :param interpolation_order:
        :type interpolation_order: int | str | None
        :param opts:
        :type opts:
        :return: interpolator
        :rtype: function
        """
        if grid.ndim == 1:
            interpolator = ProductGridInterpolator(
                grid,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif (
                grid.mesh_type == MeshType.Structured
                or grid.mesh_type == MeshType.Regular
        ):
            interpolator = ProductGridInterpolator(
                grid.subgrids,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            interpolator = UnstructuredGridInterpolator(
                grid,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif grid.mesh_type == MeshType.SemiStructured:
            raise NotImplementedError("don't know what I want to do with semistructured meshes anymore")
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return interpolator

    @classmethod
    def get_extrapolator(cls, grid, vals, extrapolation_order=1, **opts):
        """
        Returns an Extrapolator that can be called on grid points to extrapolate them

        :param grid:
        :type grid: Mesh
        :param extrapolation_order:
        :type extrapolation_order: int
        :return: extrapolator
        :rtype: Extrapolator
        """

        # Extrapolator(
        #     cls(
        #         grid,
        #         vals,
        #         interpolation_order=extrapolation_order,
        #         extrapolator=Extrapolator(lambda g: np.full(g.shape, np.nan))
        #     )
        # )

        if grid.ndim == 1:
            extrapolator = ProductGridInterpolator(
                grid,
                vals,
                order=extrapolation_order,
                extrapolate=True
            )
        elif (
                grid.mesh_type == MeshType.Structured
                or grid.mesh_type == MeshType.Regular
        ):
            extrapolator = ProductGridInterpolator(
                grid.subgrids,
                vals,
                order=extrapolation_order,
                extrapolate=True
            )
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            extrapolator = UnstructuredGridInterpolator(
                grid,
                vals,
                neighbors=extrapolation_order+1,
                order=extrapolation_order,
                extrapolate=True
            )
        elif grid.mesh_type == MeshType.SemiStructured:
            raise NotImplementedError("don't know what I want to do with semistructured meshes anymore")
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return Extrapolator(
            extrapolator,
            **opts
        )

    def apply(self, grid_points, **opts):
        """Interpolates then extrapolates the function at the grid_points

        :param grid_points:
        :type grid_points:
        :return:
        :rtype:
        """
        # determining what points are "inside" an interpolator region is quite tough
        # instead it is probably better to allow the basic thing to interpolate and do its thing
        # and then allow the extrapolator to post-process that result
        vals = self.interpolator(grid_points, **opts)
        if self.extrapolator is not None:
            vals = self.extrapolator(grid_points, vals)
        return vals

    def derivative(self, order):
        """
        Returns a new function representing the requested derivative
        of the current interpolator

        :param order:
        :type order:
        :return:
        :rtype:
        """
        return type(self)(
            self.grid,
            self.vals,
            interpolation_function=self.interpolator.derivative(order),
            extrapolator=self.extrapolator.derivative(order) if self.extrapolator is not None else self.extrapolator
        )

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

######################################################################################################
##
##                                   Extrapolator Class
##
######################################################################################################
class Extrapolator:
    """
    A general purpose that takes your data and just extrapolates it.
    This currently only exists in template format.
    """
    def __init__(self,
                 extrapolation_function,
                 warning=False,
                 **opts
                 ):
        """
        :param extrapolation_function: the function to handle extrapolation off the interpolation grid
        :type extrapolation_function: None | function | Callable | Interpolator
        :param warning: whether to emit a message warning about extrapolation occurring
        :type warning: bool
        :param opts: the options to feed into the extrapolator call
        :type opts:
        """
        self.extrapolator = extrapolation_function
        self.extrap_warning = warning
        self.opts = opts

    def derivative(self, n):
        return type(self)(
            self.extrapolator.derivative(n),
            warning=self.extrapolator
        )

    def find_extrapolated_points(self, gps, vals, extrap_value=np.nan):
        """
        Currently super rough heuristics to determine at which points we need to extrapolate
        :param gps:
        :type gps:
        :param vals:
        :type vals:
        :return:
        :rtype:
        """
        if extrap_value is np.nan:
            where = np.isnan(vals)
        elif extrap_value is np.inf:
            where = np.isinf(vals)
        elif not isinstance(extrap_value, (int, float, np.floating, np.integer)):
            where = np.logical_and(vals <= extrap_value[0], vals >= extrap_value[1])
        else:
            where = np.where(vals == extrap_value)

        return gps[where], where

    def apply(self, gps, vals, extrap_value=np.nan):
        ext_gps, inds = self.find_extrapolated_points(gps, vals, extrap_value=extrap_value)
        if len(ext_gps) > 0:
            new_vals = self.extrapolator(ext_gps)
            # TODO: emit a warning about extrapolating if we're doing so?
            vals[inds] = new_vals
        return vals

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)