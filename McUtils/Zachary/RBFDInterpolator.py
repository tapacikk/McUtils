"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""

import numpy as np, math
import scipy.spatial as spat
import itertools, typing, dataclasses
from ..Numputils import vec_outer, unique as nput_unique
from .Symbolic import TensorExpression

__all__ = [
    "RBFDInterpolator"
]

class RBFDError(ValueError):
    ...

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
                 multicenter_monomials=False,
                 neighborhood_size=15,
                 neighborhood_merge_threshold=10,
                 solve_method='least_squares',
                 max_condition_number=np.inf,
                 error_threshold=.01
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
        self.neighborhood_size = neighborhood_size
        self.merge_limit = neighborhood_merge_threshold
        self.solve_method = solve_method
        self.max_condition_number = max_condition_number # not actually using this...
        self.error_threshold = error_threshold

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
            scalings = [1 if M - m < 1e-8 else M - m for M, m in zip(maxs, mins)]
            rescaled_slices = [
                (x - m) / s for x, m, s in zip(dim_slices, mins, scalings)
            ]
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
            scaligns = self.grid_scaling.copy()
            scaligns[scaligns == 0] = 1. # only gonna happen when there's no shift anyway...
            return ((pts - self.grid_shifts[:, np.newaxis]) / scaligns[:, np.newaxis]).T
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
        # print(">>", degree, rbf_mats.shape, pol_mats.shape)
        if pol_mats is not None:
            rbf_mats = np.concatenate([rbf_mats, pol_mats], axis=1)
        if force_square and rbf_mats.shape[1] > rbf_mats.shape[0]: # force it to be square by including extra polys
            curl = rbf_mats.shape[0]
            neel = rbf_mats.shape[1]
            disp = neel-curl
            rbf_mats = np.concatenate([rbf_mats, np.zeros((disp, neel))], axis=0)
            rbf_mats[curl:, :curl] = rbf_mats[:curl, -disp:].T
        return rbf_mats

    def solve_system(self, centers, vals, derivs:list, solver=None, return_data=False, error_threshold=None):
        if len(vals) != len(centers):
            raise ValueError("need same number of interpolation points and values (got {} and {})".format(
                len(centers),
                len(vals)
            ))
        # TODO: compute the appropriate deriv order to match the vals and derivs
        derivs = [d.flatten() for d in derivs]
        nder = len(derivs)
        og_vals = vals
        vals = np.concatenate([vals]+derivs, axis=0)

        degree = 0
        if nder > 0:
            # I now know how many derivs I have per order...but I get the same contrib from each
            # degree order, but the issue is the num derivs scales with the number of centers too
            # so it needs to be solved for differently
            ndim = centers.shape[-1]
            target_extra = len(centers)*sum(self.triu_num(ndim, k) for k in range(1, nder + (0 if self.multicenter_monomials else 1) )) - 1
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
        cn = np.linalg.cond(M)
        if cn > self.max_condition_number:
            print("WARNING: condition number={}".format(cn))

        if solver is None:
            solver = self.solve_method
        if solver == "least_squares":
            w_data = np.linalg.lstsq(M, vals, rcond=None)
        else:
            w_data = np.linalg.solve(M, vals)#, rcond=None)
            w_data = (w_data,)

        test_vals = np.dot(M, w_data[0])
        nv = len(og_vals)
        base_error = og_vals - test_vals[:nv]
        # often there's a consistent shift when things break down
        extra_shift = np.average(base_error)

        evec = np.concatenate([
            base_error - extra_shift,
            (vals[nv:] - test_vals[nv:])
        ])
        test_norm = np.linalg.norm(evec) / len(evec)  # average percent error ?
        # print(">>", test_norm, error_threshold)
        # print(len(evec), len(vals))

        test_err = np.max(np.abs(evec)) # max diff error?

        errors = [test_norm, test_err]

        if error_threshold is None:
            error_threshold = self.error_threshold

        if error_threshold is not None:
            if test_norm > error_threshold: #
                # print("...?")
                raise RBFDError("solution doesn't solve equations")
                # print("="*5)
                # print(test_norm)
                # print(vals - test_vals)



        if return_data:
            data = {'matrix':M, 'vals':vals, 'condition_number':cn}
            return w_data, degree+self.extra_degree, extra_shift, errors, data
        else:
            return w_data, degree+self.extra_degree, extra_shift, errors

    def get_neighborhood(self, pts, *, neighbors, return_distances=False):
        if neighbors is None:
            neighbors = self.neighborhood_size
        distances, yindices = self.tree.query(pts, k=neighbors)
        if neighbors == 1:
            # `KDTree` squeezes the output when neighbors=1.
            yindices = yindices[np.newaxis]
        # yindices = np.sort(yindices)
        res = yindices
        if return_distances:
            res = (yindices, distances)
        return res

    def create_neighbor_groups(self, inds, merge_limit=None):
        #TODO: lots of optimizations here, cutting down on loops, making use of sorting
        #      not doing useless setdiffs, etc.
        # _, sorting, subinds = nput_unique(inds, axis=0, return_index=True)
        if merge_limit is None:
            merge_limit = self.merge_limit
        inds, sorting, counts = nput_unique(inds, axis=0, return_counts=True)
        subinds = np.split(sorting, np.cumsum(counts))[:-1]
        # raise Exception(nput_unique(inds, axis=0, return_index=True))
        if merge_limit is not None:
            allowed_diffs = len(inds[0]) - merge_limit
        else:
            allowed_diffs = -1
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
        __slots__ = ['weights', 'centers', 'degree', 'scaling_data', 'extra_shift', 'interpolation_error', 'solver_data']
        def __init__(self, w, grid, degree, scaling_data, extra_shift=0, interpolation_error=0, solver_data=None):
            self.weights = w
            self.centers = grid
            self.degree = degree
            self.scaling_data = scaling_data
            self.solver_data = solver_data
            self.extra_shift = extra_shift
            self.interpolation_error = interpolation_error
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

    def apply_interpolation(self, pts, data, reshape_derivatives=True, return_matrix=False, deriv_order=0):
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
        vals = scaling_data.reverse_value_renormalization(v[:npts] - data.extra_shift)
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

        res = (vals, dvs)
        if return_matrix:
            res = res + (M,)
        return res

    def construct_interpolation(self, inds, solver_data=False, return_error=False):
        grid = self.grid[inds]
        vals = self.vals[inds]
        derivs = [d[inds] for d in self.derivs]

        # print("=>", grid.shape, vals.shape, [d.shape for d in derivs])

        grid, vals, derivs, scaling_data = self.RescalingData.initialize_subgrid_data(
            grid,
            vals,
            derivs
        )

        # print("< ", grid.shape, vals.shape, [d.shape for d in derivs])

        res = self.solve_system(grid, vals, derivs, return_data=solver_data)
        if solver_data:
            w, degree, extra, error, data = res
        else:
            w, degree, extra, error = res
            data = None

        new = self.InterpolationData(w, grid, degree, scaling_data, extra_shift=extra, interpolation_error=error, solver_data=data)
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

    def nearest_interpolation(self, pts, neighbors=None, solver_data=False, interpolator=True):
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

    def eval(self, pts, deriv_order=0, neighbors=None,
             merge_neighbors=None,
             reshape_derivatives=True,
             return_interpolation_data=False,
             check_in_sample=True,
             zero_tol=1e-8,
             return_error=False,
             use_cache=True,
             retries=3
             # extrapolation_warning=None
             ):
        pts = np.asanyarray(pts)
        if pts.ndim == 1:
            pts = pts[np.newaxis]

        if pts.ndim > 2:
            extra_shape = pts.shape[:-1]
            pts = pts.reshape(-1, pts.shape[-1])
        else:
            extra_shape = None

        # set up return locations...
        val_sets = np.empty(len(pts))
        interp_data = np.full(len(pts), None) if return_interpolation_data else None
        der_sets = [
                    np.empty((len(pts),) + d.shape[1:])
                    for d in self.derivs
                ] if deriv_order > 0 else None
        errors = np.empty((len(pts), 2)) if return_error else None

        hoods, dists = self.get_neighborhood(pts, neighbors=neighbors, return_distances=check_in_sample)

        sub_map = None # to map from the reduced sample to the full one if we are able to reduce
        if check_in_sample: # look for things very close to an abcissa
            zero_pos = np.where(dists < zero_tol)
            if len(zero_pos) > 0:
                sample_pos = hoods[zero_pos]
                zero_pos = zero_pos[0]
                if len(zero_pos) > 0:
                    val_sets[zero_pos] = self.vals[sample_pos]
                    for n,d in enumerate(self.derivs):
                        der_sets[n][zero_pos] = d[sample_pos]
                    if return_error:
                        errors[zero_pos] = np.broadcast_to([[0, 0]], (len(zero_pos), 2))
                    sub_map = np.setdiff1d(np.arange(len(pts)), zero_pos)
                    hoods = hoods[sub_map]
                    pts = pts[sub_map]

        ind_grps, pts_grps = self.create_neighbor_groups(hoods, merge_limit=merge_neighbors)

        # print(ind_grps, pts_grps)
        cache = self._pos_cache if use_cache else {}
        for pts_inds, vals_inds in zip(pts_grps, ind_grps): # we need to evaluate this entire loop for each point......
            k = tuple(vals_inds)
            if k not in cache:
                neighs = self.neighborhood_size if neighbors is None else neighbors
                et = self.error_threshold
                for n in range(retries):
                    try:
                        if n == retries - 1:
                            self.error_threshold = None # always gotta succeed in the end...
                        interp_stuff = self.construct_interpolation(vals_inds, solver_data=return_interpolation_data, return_error=return_error)
                    except RBFDError:
                        # print(">>", n, vals_inds)
                        neighs = neighs * 2 # just try to expand a little bit to fix things...
                        new_nearest = self.get_neighborhood(pts[pts_inds], neighbors=neighs)
                        vals_inds = np.unique(np.concatenate([vals_inds, new_nearest.flatten()]))
                    else:
                        break
                    finally:
                        self.error_threshold = et
                if return_error:
                    errors[pts_inds, :] = np.broadcast_to(interp_stuff.interpolation_error, (len(pts_inds), 2))
                cache[k] = interp_stuff
            res = self.apply_interpolation(pts[pts_inds], cache[k], deriv_order=deriv_order,
                                           reshape_derivatives=reshape_derivatives, return_matrix=return_interpolation_data
                                           )
            res_inds = pts_inds if sub_map is None else sub_map[pts_inds]
            if return_interpolation_data:
                vals, ders, mat = res
                interp_data[res_inds] = {
                    'points':self.grid[vals_inds],
                    'data': cache[k],
                    'matrix': mat,
                    'solver_data': cache[k].solver_data
                }
            else:
                vals, ders = res
            val_sets[res_inds] = vals
            for n, d in enumerate(ders):
                der_sets[n][res_inds] = d

        if extra_shape is not None:
            val_sets = np.reshape(val_sets, extra_shape)
            if return_interpolation_data:
                interp_data = np.reshape(interp_data, extra_shape)
            if return_error:
                errors = np.reshape(errors, extra_shape +(2,))
            der_sets = [
                d.reshape(extra_shape + d.shape[1:])
                for d in der_sets
            ]

        if deriv_order > 0:
            if der_sets is None and len(pts) == 0:
                der_sets = []
            res = [val_sets] + der_sets
        else:
            res = val_sets

        if return_interpolation_data or return_error:
            res = (res,)

        if return_error:
            res = res + (errors,)

        if return_interpolation_data:
            res = res + (interp_data,)

        return res

    def __call__(self, pts, deriv_order=0, neighbors=None, merge_neighbors=None, reshape_derivatives=True,
                 return_interpolation_data=False,
                 use_cache=True,
                 return_error=False
                 ):
        return self.eval(pts,
                         deriv_order=deriv_order, neighbors=neighbors,
                         merge_neighbors=merge_neighbors, reshape_derivatives=reshape_derivatives,
                         return_interpolation_data=return_interpolation_data,
                         use_cache=use_cache,
                         return_error=return_error
                         )

    @property
    def global_interpolator(self):
        if self._globint is None:
            et = self.error_threshold
            try: # ignore errors
                self.error_threshold = None
                interp = self.construct_interpolation(np.arange(len(self.grid)))
            finally:
                self.error_threshold = et
            self._globint = self.Interpolator(interp, self)
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