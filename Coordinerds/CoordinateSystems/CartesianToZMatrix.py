from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinateSystem, ZMatrixCoordinateSystem
from ...Numputils import vec_norms, vec_angles, pts_dihedrals
import numpy as np
# this import gets bound at load time, so unfortunately PyCharm can't know just yet
# what properties its class will have and will try to claim that the files don't exist

class CartesianToZMatrixConverter(CoordinateSystemConverter):
    """A converter class for going from Cartesian coordinates to ZMatrix coordinates

    """

    @property
    def types(self):
        return (CartesianCoordinateSystem, ZMatrixCoordinateSystem)

    def canonicalize_order_list(self, ncoords, order_list):
        """Normalizes the way the ZMatrix coordinates are built out

        :param ncoords:
        :type ncoords:
        :param order_list: the basic ordering to apply for the
        :type order_list: iterable or None
        :return:
        :rtype: iterator of int triples
        """
        if order_list is None:
            normalized_list = np.array( (
                   np.arange(ncoords),
                   np.arange(-1, ncoords-1),
                   np.arange(-2, ncoords-2),
                   np.arange(-3, ncoords-3),
                ) ).T
        else:
            normalized_list = [None] * len(order_list)
            for i, el in enumerate(order_list):
                if isinstance(el, int):
                    spec = (
                        el,
                        normalized_list[i-1][0] if i > 0 else -1,
                        normalized_list[i-2][0] if i > 1 else -1,
                        normalized_list[i-3][0] if i > 2 else -1
                    )
                else:
                    spec = tuple(el)
                    # if len(spec) < 4:
                    #     spec = (i,) + spec + (
                    #     normalized_list[i-1][0] if i > 0 else -1,
                    #     normalized_list[i-2][0] if i > 1 else -1,
                    #     normalized_list[i-3][0] if i > 2 else -1
                    # )
                    # spec = spec[:4]
                    if len(spec) < 4:
                        raise ValueError(
                            "Z-matrix conversion spec {} not long enough. Expected ({}, {}, {}, {})".format(
                                el,
                                "atomNum", "distAtomNum", "angleAtomNum", "dihedAtomNum"
                            ))

                normalized_list[i] = spec
        return np.asarray(normalized_list, dtype=np.int8)

    @staticmethod
    def get_dists(points, centers):
        return vec_norms(centers-points)
    @staticmethod
    def get_angles(lefts, centers, rights):
        # need to look up again what the convention is for which atom is the central one...
        v1s = centers-lefts
        v2s = centers-rights
        return vec_angles(v1s, v2s)[0]
    @staticmethod
    def get_diheds(points, centers, seconds, thirds):
        return pts_dihedrals(points, centers, seconds, thirds)

    def convert_many(self, coords, ordering = None, use_rad=True, **kw):
        """
        We'll implement this by having the ordering arg wrap around in coords?
        """
        if ordering is None:
            ordering = range(len(coords[0]))
        base_shape = coords.shape
        new_coords = np.reshape(coords, (np.product(base_shape[:-1]),) + base_shape[-1:])
        new_coords, ops = self.convert(new_coords, ordering=ordering, use_rad=use_rad)
        new_shape = base_shape[:-2] + (base_shape[-2]-1, new_coords.shape[-1])
        new_coords = np.reshape(new_coords, new_shape)
        return new_coords, ops

    def convert(self, coords, ordering=None, use_rad=True, **kw):
        """The ordering should be specified like:

        [
            [n1],
            [n2, n1]
            [n3, n1/n2, n1/n2]
            [n4, n1/n2/n3, n1/n2/n3, n1/n2/n3]
            [n5, ...]
            ...
        ]

        :param coords:    array of cartesian coordinates
        :type coords:     np.ndarray
        :param use_rad:   whether to user radians or not
        :type use_rad:    bool
        :param ordering:  optional ordering parameter for the z-matrix
        :type ordering:   None or tuple of ints or tuple of tuple of ints
        :param kw:        ignored key-word arguments
        :type kw:
        :return: z-matrix coords
        :rtype: np.ndarray
        """
        ncoords = len(coords)
        orig_ol = self.canonicalize_order_list(ncoords, ordering)
        ol = orig_ol
        nol = len(ol)

        multiconfig = nol < ncoords
        if multiconfig:
            fsteps = ncoords / nol
            steps = int(fsteps)
            if steps != fsteps:
                raise ValueError(
                    "{}: Number of coordinates {} and number of specifed elements {} misaligned".format(
                        type(self),
                        ncoords,
                        nol
                    )
                )
            # broadcasts a single order spec to be a multiple order spec
            ol = np.reshape(
                np.broadcast_to(ol, (steps, nol, 4)) +
                np.reshape(np.arange(0, ncoords, nol), (steps, 1, 1)),
                (ncoords, 4)
            )


        # we define an order map that we'll index into to get the new indices for a
        # given coordinate
        om = 1+np.argsort(ol[:, 0])

        # need to check against the cases of like 1, 2, 3 atom molecules
        # annoying but not hard
        if not multiconfig:
            dists = self.get_dists(
                coords[ol[1:, 0]],
                coords[ol[1:, 1]]
            )
            if len(ol) > 2:
                angles = np.concatenate( (
                    [0],
                    self.get_angles(
                        coords[ol[2:, 0]],
                        coords[ol[2:, 1]],
                        coords[ol[2:, 2]]
                    )
                ) )
                if not use_rad:
                    angles = np.rad2deg(angles)
            else:
                angles = np.array([0.])
            if len(ol) > 3:
                diheds = np.concatenate( (
                    [0, 0],
                    self.get_diheds(
                        coords[ol[3:, 0]],
                        coords[ol[3:, 1]],
                        coords[ol[3:, 2]],
                        coords[ol[3:, 3]]
                    )
                ) )
                if not use_rad:
                    diheds = np.rad2deg(diheds)
            else:
                diheds = np.array([0, 0])
            ol = ol[1:]

        else: # multiconfig

            # we do all of this stuff with masking operations in the multiconfiguration cases
            mask = np.repeat(True, ncoords)
            mask[np.arange(0, ncoords, nol)] = False
            dists = self.get_dists(
                coords[ol[mask, 0]],
                coords[ol[mask, 1]]
            )

            if nol>2:
                # set up the mask to drop all of the first bits
                mask[np.arange(1, ncoords, nol)] = False
                angles = self.get_angles(
                    coords[ol[mask, 0]],
                    coords[ol[mask, 1]],
                    coords[ol[mask, 2]]
                )
                angles = np.append(angles, np.zeros(steps))
                insert_pos = np.arange(0, ncoords-1*steps-1, nol-2)
                angles = np.insert(angles, insert_pos, 0)
                angles = angles[:ncoords-steps]
                if not use_rad:
                    angles = np.rad2deg(angles)
            else:
                angles = np.zeros(ncoords-steps)


            if nol > 3:
                # set up mask to drop all of the second bits (wtf it means 'second')
                mask[np.arange(2, ncoords, nol)] = False
                diheds = self.get_diheds(
                    coords[ol[mask, 0]],
                    coords[ol[mask, 1]],
                    coords[ol[mask, 2]],
                    coords[ol[mask, 3]]
                )
                # pad diheds to be the size of ncoords
                diheds = np.append(diheds, np.zeros(2*steps))

                # insert zeros where undefined
                diheds = np.insert(diheds, np.repeat(np.arange(0, ncoords-2*steps-1, nol-3), 2), 0)
                # take only as many as actually used
                diheds = diheds[:ncoords-steps]
                if not use_rad:
                    diheds = np.rad2deg(diheds)
            else:
                diheds = np.zeros(ncoords-steps)


            # after the np.insert calls we have the right number of final elements, but too many
            # ol and om elements and they're generally too large
            # so we need to shift them down and mask out the elements we don't want
            mask = np.repeat(True, ncoords)
            mask[np.arange(0, ncoords, nol)] = False
            ol = np.reshape(ol[mask], (steps, nol-1, 4))-np.reshape(np.arange(steps), (steps, 1, 1))
            ol = np.reshape(ol, (ncoords-steps, 4))
            om = np.reshape(om[mask], (steps, nol-1))-nol*np.reshape(np.arange(steps), (steps, 1))-1
            om = np.reshape(om, (ncoords-steps,))

        #### should find some way to return the order ?
        final_coords = np.array(
            [
                dists, angles, diheds
            ]
        ).T


        if multiconfig:
            # figure out what to use for the axes
            origins = coords[ol[::nol-1, 1]] # whatever was referenced by the first atom in the spec
            x_axes  = coords[ ol[::nol-1, 0]] - origins # the first displacement vector
            y_axes  = coords[ ol[1::nol-1, 0]] - origins # the second displacement vector (just defines the x-y plane, not the real y-axis)
            axes = np.array([x_axes, y_axes]).transpose((1, 0, 2))
            # print(origins.shape, axes.shape)
            # print(origins)
            # print(axes)

        else:
            axes = np.array([coords[1] - coords[0], coords[2] - coords[0]])
            origins = coords[0]

        ol = orig_ol
        om = om - 1
        ordering = np.array(
                [
                    np.arange(len(ol)), om[ol[:, 1]], om[ol[:, 2]], om[ol[:, 3]]
                ]
            ).T

        return final_coords, dict(use_rad=use_rad, ordering=ordering, origins=origins, axes=axes)

__converters__ = [ CartesianToZMatrixConverter() ]