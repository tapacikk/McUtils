"""
Experimental module that provides a FiniteDifferenceDerivative class that does finite-difference derivatives
"""

from .FiniteDifferenceFunction import FiniteDifferenceFunction
import numpy as np, itertools as it

__all__ = [
    'FiniteDifferenceDerivative'
]

class FiniteDifferenceDerivative:
    """Provides derivatives for a function (scalar or vector valued)
    Can be indexed into or the entire tensor of derivatives may be requested
    """
    def __init__(self, f, order = 1, function_shape = (0, 0), **fd_opts):
        """

        :param f: the function we would like to take derivatives of
        :type f: callable
        :param order: the order of the derivative we would like to take
        :type order: int | list[int]
        :param function_shape: The input and output shapes of f. None means a scalar function.
        :type function_shape: int | iterable[iterable[int] | int]
        :param fd_opts: the options to pass to the finite difference function
        :type fd_opts:
        """
        self.f = f
        self.n = order
        self._fd_opts = fd_opts
        self._f_shape = function_shape
        self._fd = None

    class differencer:
        """
        The lower-level class that takes the entire high-level spec and converts it into derivatives.
        Allows indexing for memory efficiency
        """
        def __init__(self,
                     f,
                     order, center, coordinates,
                     displacement_function,
                     input_shape, output_shape,
                     mesh_spacing, prep,
                     **fd_opts
                     ):

            # set up the coordinate parameters
            if isinstance(center, (int, float, np.integer, np.float)):
                center = [center]
            center = np.asarray(center)

            # determine if center is multiple configurations or just one
            multiconfig = len(center.shape) > (0 if isinstance(input_shape, (int, np.integer)) else len(input_shape))
            if not multiconfig:
                coords = np.array([center])
            else:
                coords = np.asarray(center)
            if len(coords.shape) == 1:
                coords = coords.reshape(coords.shape + (1,))

            # determine up the data shapes
            shape = coords[0].shape
            coord_shape = tuple(shape if isinstance(input_shape, (int, np.integer)) else input_shape)

            # figure out how many dimensions we will have flattened for a multiconfiguration state
            flattened_dims = None
            if len(shape) > len(coord_shape):
                flattened_dims = coords.shape[: len(coords.shape) - len(coord_shape)]
                num_flattened = np.product(flattened_dims)
                coords = coords.reshape((num_flattened, ) + coord_shape)

            # configure the displacements
            if displacement_function is None:
                displacement_function = lambda a:a
            displacement = displacement_function(mesh_spacing) # provides a default displacement to work with
            if isinstance(displacement, (float, np.float)):
                displacement = np.full(coord_shape, displacement)
            elif displacement.shape == coord_shape[-1:]:
                displacement = np.broadcast_to(displacement, coord_shape)

            # create the coordinate specs to generate derivatives over
            if coordinates is not None and coordinates[0] is not None:
                to_gen = coordinates[0]
            else:
                to_gen = np.arange(np.product(np.array(coord_shape)))
            if isinstance(to_gen[0], (int, np.integer)): # willing to bet all these -2 things are wrong...
                to_gen = np.array(np.meshgrid(*([to_gen] * order)))
                to_gen = np.array(np.unravel_index(to_gen, coord_shape)).transpose()
                to_gen = to_gen.reshape((np.product(to_gen.shape[:-2]), ) + to_gen.shape[-2:] )

            # create the coordinate specs to generate the derivatives of (only makes sense for non-scalar functions)
            if coordinates is not None and coordinates[1] is not None:
                gen_partners = tuple(coordinates[1])
            else:
                gen_partners = None

            # define a function to prep the system
            if prep is None:
                prep = lambda c, a, b: (a, b)


            self.f = f
            self.order = order
            self.displacement = displacement
            self.coords = coords
            self.to_gen = to_gen
            self.gen_partners = gen_partners
            self.multiconfig = multiconfig
            self.prep = prep
            self.fd_opts = fd_opts
            self.in_shape = coord_shape
            self.out_shape = output_shape
            self._flattend_dims = flattened_dims

        def _get_displaced_coords(self, coord, stencil_widths, stencil_shapes):

            coords = self.coords
            displacement = self.displacement

            # not too worried about looping over coordinates since the number of loops will be like max in the small hundreds
            num_displacements = np.product(stencil_widths)
            displacement_shape = (num_displacements, ) + coords.shape[1:]

            displacements = np.zeros(displacement_shape) # fuck ton of zeros
            base_roll = tuple(np.arange(len(stencil_widths)))

            coord = np.unique(coord, axis=0)

            for i, sc in enumerate(zip(stencil_shapes, coord)):
                stencil_shape, c = sc
                # coord can be like ((1, 1), (1, 2)) or ((0,), (1,)) in which case we have a 2D derivative
                # creates single displacement matrix
                ctup = tuple(c)

                disp = displacement[ctup]

                steps = np.arange(-(stencil_shape[0]-1), stencil_shape[1]+1)
                full_disp = disp * steps
                to_set = np.broadcast_to(full_disp, stencil_widths).transpose(np.roll(base_roll, -2+i)).flatten()

                # in dimension 1 we want to have this repeat over the slowest moving indices, then the ones after those
                # then after those, etc.
                idx = (...,) + ctup
                # I'm imposing some assumptions on the form of this for the moment, but we can relax those later...

                displacements[idx] = to_set

            # then we broadcast *this* up to the total number of walkers we have
            full_target_shape = (len(coords), ) + displacement_shape
            coords_expanded = np.expand_dims(coords, 1)
            displaced_coords = np.broadcast_to(coords_expanded, full_target_shape) # creates the displaced coordinate sets
            full_full_displacement = np.broadcast_to(displacements, full_target_shape)

            return displacements, displaced_coords + full_full_displacement

        @staticmethod
        def _get_diff(c, disp):
            diffs = np.abs(np.diff(disp[(..., ) + tuple(c)]))
            return np.sort(diffs)[-1] # get the only diff > 0 (assumes we have a regular grid)

        _fdfs = {}
        def _get_fdf(self, ci):
            # create the different types of finite differences we'll compute for the different coordinates of interest
            dorder = self._dorder(ci)
            try:
                fdf = self._fdfs[dorder]
            except KeyError:
                fdf = FiniteDifferenceFunction.RegularGridFunction(
                        dorder,
                        end_point_precision = 0,
                        only_core = False,
                        only_center = True,
                        # axis = 1,
                        **self.fd_opts
                    )

            stencil_widths = tuple( len(cf[1]) if cf is not None else cf for cf in fdf.coefficients )
            stencil_shapes = tuple( w[1] if w is not None else w for w in fdf.widths )
            finite_difference = fdf.get_FDF(shape = stencil_widths)

            return stencil_widths, stencil_shapes, finite_difference, fdf,dorder

        def _coord_index(self, coord):
            # gotta re-linearize coord so we can use it to index into like fdfs and stuff...
            raveled = [ np.ravel_multi_index(c, self.in_shape) for c in coord ]
            return raveled

        def _dorder(self, raveled):
            unique, counts = np.unique(raveled, return_counts=True)
            pairs = np.array([unique, counts]).T
            orders = np.sort(pairs, axis=0)[:, 1]
            return tuple(orders)

        def coordinate_derivs(self, coord, coordinates = None, pull_center = True, return_coords = False):
            """
            Computes the derivatives for the function values given by coordinates with respect to the coordinate coord

            :param coord: the coordinate to take derivatives w.r.t
            :type coord: iterable[iterable[int]]
            :param coordinates: the coordinates for the function output to use in the derivatives
            :type coordinates: iterable[int]
            :return:
            :rtype:
            """
            f = self.f
            order = self.order
            out_shape = self.out_shape

            ci = self._coord_index(coord)
            stencil_widths, stencil_shapes, finite_difference, fdf, dorder = self._get_fdf(ci)

            if coordinates is None:
                coordinates = self.gen_partners

            displacements, displaced_coords = self._get_displaced_coords(coord, stencil_widths, stencil_shapes)
            function_values = f(displaced_coords)

            disp, fvals = self.prep(coord, displacements, function_values)

            # fvals is for _all_ of the configurations so we need to reshape it based on the out_shape
            # we want it to be shaped such that we have the fvals for each configuration in an array
            # for this we'll finally make use of the out_shape parameter
            o_shape = (1,) if isinstance(out_shape, (int, np.integer)) else out_shape
            if len(fvals.shape) < 2 + len(o_shape): # is this the right heuristic...?
                fvals = fvals.reshape(fvals.shape + (1,)* ( 2 + len(o_shape) - len(fvals.shape) ) )
            # we use this to flatten our system so that all of our potentially n-D fvals are linearized
            num_coords = np.product(o_shape)
            new_coord_shape = fvals.shape[:-1] + (num_coords, )

            fvals = fvals.reshape(new_coord_shape)

            # now we potentially filter out some coords...
            if coordinates is not None:
                fvals = fvals[..., coordinates]



            # finally we restructure this into a tensor of the appropriate dimension for feeding into the FD code
            fvals_shape = (len(function_values), ) + stencil_widths + fvals.shape[-1:]
            fvals = fvals.reshape(fvals_shape)

            h = [ self._get_diff(c, disp) for c in coord ]

            derivs = finite_difference(fvals, axis = 1, h = h)

            if pull_center:
                # the finite difference code will give us more derivatives than the one we actually care about, generally
                # it will, e.g., give us derivatives at x-2, x-1, x+1, and x+2 in a 5-point stencil
                # so we only pull the *middle* derivatives, even though we've clearly wasted some work here.
                dorder = len(stencil_widths)
                deriv_center = tuple( int(np.floor(s/2)) for s in derivs.shape[1:dorder+1] )
                schloop = (slice(None, None, None),) + deriv_center
                derivs = derivs[schloop]
            else:
                roll = tuple(np.roll(np.arange(order + 1), -1)) + tuple(np.arange(dorder + 1, len(derivs.shape)))
                derivs = np.transpose(derivs, roll)

            if return_coords:
                return displaced_coords, derivs
            else:
                return derivs

        def compute_derivatives(self, coords = None, others = None, pull_center = True, return_coords = False):
            """Computes the derivatives for others with respect to coords

            :param coords:
            :type coords:
            :param others:
            :type others:
            :param pull_center:
            :type pull_center:
            :return:
            :rtype:
            """
            from operator import itemgetter
            from functools import reduce

            to_gen = self.to_gen if coords is None else coords
            gen_partners = self.gen_partners if others is None else others
            coords = self.coords
            order = self.order
            coord_shape = self.in_shape

            full_tensor = None
            to_gen = np.array(reduce(lambda a, k:sorted(a, key=itemgetter(k)), reversed(range(order)), to_gen))

            to_gen_unique = tuple( np.unique( to_gen[:, i], axis=0 ) for i in range(order) )
            base_tensor_shape = tuple( len(u) for u in to_gen_unique )

            for coord_num, coord in enumerate(to_gen):

                if return_coords: # not doing anything with this yet...
                    coords, derivs = self.coordinate_derivs(coord,
                                                            coordinates=gen_partners,
                                                            pull_center=pull_center,
                                                            return_coords = return_coords
                                                            )
                else:
                    derivs = self.coordinate_derivs(coord,
                                                    coordinates=gen_partners,
                                                    pull_center=pull_center,
                                                    return_coords = return_coords
                                                    )
                #TODO: Need to figure out how I'm gonna reconcile having different types of derivatives
                # in here, particularly things like dx2 and dxidyi

                if full_tensor is None: # we finally know what size it'll be...
                    full_tensor_shape = (len(coords), ) + base_tensor_shape + derivs.shape[1:]
                    full_tensor = np.zeros(full_tensor_shape)

                coord_pos = np.unravel_index(coord_num, base_tensor_shape)
                for config_num in range(len(coords)): # a python loop but hopefully a relatively cheap one...
                    set_index = (config_num, ) + coord_pos
                    full_tensor[set_index] = derivs[config_num]

            if not self.multiconfig:
                full_tensor = full_tensor[0]

            if isinstance(self.out_shape, (int, np.integer)):
                if pull_center:
                    full_tensor = full_tensor.reshape(full_tensor.shape[:-1])
                else:
                    full_tensor = full_tensor.reshape((full_tensor.shape[0], full_tensor.shape[2]))

            if self._flattend_dims is not None:
                full_tensor = full_tensor.reshape(self._flattend_dims + full_tensor.shape[1:])

            return full_tensor

        @property
        def array(self):
            """Computes the entire tensor of derivatives based on the provided spec

            :return:
            :rtype:
            """

            return self.compute_derivatives()

        def _idx(self, c, coord_shape = None):
            if coord_shape is None:
                coord_shape = self.in_shape
            return np.array(np.unravel_index(np.array(c), coord_shape)).T
        def _fidx(self, c, coord_shape = None):
            if coord_shape is None:
                coord_shape = self.in_shape
            return np.ravel_multi_index(c, coord_shape)

        @property
        def shape(self):
            ip = np.product(self.in_shape)
            op = np.product(self.out_shape)
            return (ip,) * self.order + (op,)

        def _get_block_derivs(self, block):
            """Pulls a block of derivatives by figuring out which coordinate indices we actually need derivs for"""
            import itertools as it

            idx = self._idx(block)

            order = self.order - len(idx)
            coord_shape = self.in_shape
            to_gen = range(np.product(coord_shape))

            to_gen = [ list(it.chain(idx, np.array([t]).T)) for t in it.combinations(to_gen, order) ]

            return self.compute_derivatives(coords = to_gen)
            # would use a generator but I think I call len on it...

        def _get_single_deriv(self, idx):

            to_gen = self._idx(idx[:-1])
            which = [ idx[-1] ]
            derivs = self.coordinate_derivs(to_gen, which)
            if self._flattend_dims is not None:
                derivs = derivs.reshape(self._flattend_dims + derivs.shape[1:])

            return derivs

        def __getitem__(self, item):
            # currently I only support tuples of indices
            if isinstance(item, (int, np.integer)):
                return self._get_block_derivs((item,))
            elif len(item) == self.order + 1:
                return self._get_single_deriv(item)
            else:
                return self._get_block_derivs(item)

    @property
    def array(self):
        return self.derivatives().array

    def derivatives(self,
                    center = 0, order = None, coordinates = None,
                    displacement_function = None, mesh_spacing = .01, prep = None,
                    **fd_opts
                    ):

        """Generates a differencer object that can be used to get derivs however your little heart desires

        :param center:
        :type center:
        :param order:
        :type order:
        :param coordinates:
        :type coordinates:
        :param displacement_function:
        :type displacement_function:
        :param mesh_spacing:
        :type mesh_spacing:
        :param prep:
        :type prep:
        :param fd_opts:
        :type fd_opts:
        :return:
        :rtype:
        """

        f = self.f
        if order is None:
            order = self.n
        input_shape, output_shape = self._f_shape
        fd_opts = dict(self._fd_opts, **fd_opts)

        return self.differencer(
            f,
            order, center, coordinates,
            displacement_function,
            input_shape, output_shape,
            mesh_spacing, prep,
            **fd_opts
        )



