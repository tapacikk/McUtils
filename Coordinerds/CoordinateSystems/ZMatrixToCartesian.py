from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinateSystem, ZMatrixCoordinateSystem
from ...Numputils import vec_norms, vec_angles, pts_dihedrals, vec_normalize, vec_crosses, \
    affine_matrix, merge_transformation_matrix, rotation_matrix, affine_multiply
import numpy as np
# this import gets bound at load time, so unfortunately PyCharm can't know just yet
# what properties its class will have and will try to claim that the files don't exist

class ZMatrixToCartesianConverter(CoordinateSystemConverter):
    """A converter class for going from ZMatrix coordinates to Cartesian coordinates

    """

    @property
    def types(self):
        return (ZMatrixCoordinateSystem, CartesianCoordinateSystem)

    @staticmethod
    def zmatrix_affine_transforms(centers, vecs1, vecs2, angles, dihedrals):
        """Builds a single set of affine transformation matrices to apply to the vecs1 to get the next set of points

        :param centers: central coordinates
        :type centers: np.ndarray
        :param vecs1: vectors coming off of the centers
        :type vecs1: np.ndarray
        :param vecs2: vectors coming off of the centers
        :type vecs2: np.ndarray
        :param angles: angle values
        :type angles: np.ndarray
        :param dihedrals: dihedral values
        :type dihedrals: np.ndarray | None
        :return:
        :rtype:
        """
        crosses = vec_crosses(vecs2, vecs1)
        rot_mats_1 = rotation_matrix(crosses, angles)
        if dihedrals is not None:
            rot_mats_2 = rotation_matrix(vecs1, -dihedrals)
            rot_mat = np.matmul(rot_mats_2, rot_mats_1)
        else:
            rot_mat = rot_mats_1
        transfs = affine_matrix(rot_mat, centers)

        return transfs

    def build_next_points(self, refs1, dists, refs2, angles, refs3, dihedrals, ref_axis = None):
        vecs1 = dists*vec_normalize(refs2 - refs1)
        if dihedrals is not None:
            vecs2 = refs3 - refs1
        else:
            # if no dihedral, for now we'll let our new axis be some random things in the x-y plane
            if ref_axis is None:
                vecs2 = np.concatenate((np.random.uniform(.5, 1, (len(refs1), 2)), np.zeros((len(refs1), 1))), axis=-1)
            elif ref_axis.ndim == 1:
                vecs2 = np.broadcast_to(ref_axis[np.newaxis, :], (len(refs1), 2))
            else:
                vecs2 = ref_axis
        transfs = self.zmatrix_affine_transforms(refs1, vecs1, vecs2, angles, dihedrals)
        newstuff = affine_multiply(transfs, vecs1)
        return newstuff

    def default_ordering(self, coordlist):
        if coordlist.shape[-1] == 6:
            ordering = coordlist[:, :, (0, 2, 4)]
            coordlist = coordlist[:, :, (1, 3, 5)]
        else:
            r = np.arange(len(coordlist[0]))
            ordering = np.broadcast_to(
                np.array([r, np.roll(r, 1), np.roll(r, 2)]).T[np.newaxis],
                coordlist.shape[:2] + (3,)
            )
        return ordering, coordlist

    def _derivs(self, coordlist, ordering=None, origins=None, axes=None, use_rad=True, **kw):
        """
        Returns the derivatives of the generated Cartesians with respect to the coordlist
        Uses analytic expressions

        :param coordlist:
        :type coordlist: np.ndarray
        :param ordering:
        :type ordering:
        :param origins:
        :type origins:
        :param axes:
        :type axes:
        :param use_rad:
        :type use_rad:
        :param kw:
        :type kw:
        :return:
        :rtype:
        """



    def convert_many(self, coordlist,
                     ordering=None, origins=None, axes=None, use_rad=True,
                     return_derivs = False,
                     **kw
                     ):
        """Expects to get a list of configurations
        These will look like:
            [
                [dist, angle, dihedral]
                ...
            ]
        and ordering will be
            [
                [pos, point, line, plane]
                ...
            ]
        **For efficiency it is assumed that all configurations have the same length**

        :param coordlist:
        :type coordlist:
        :param origins:
        :type origins:
        :param axes:
        :type axes:
        :param use_rad:
        :type use_rad:
        :param kw:
        :type kw:
        """
        # make sure we have the ordering stuff in hand
        if ordering is None:
           ordering, coordlist = self.default_ordering(coordlist)

        orderings = ordering
        if np.min(orderings) > 0:
            orderings = orderings - 1
        dim_diff = coordlist.ndim - orderings.ndim
        if dim_diff > 0:
            missing = coordlist.shape[:dim_diff]
            orderings = np.broadcast_to(ordering, missing + orderings.shape )

        if orderings.shape[-1] == 4:
            atom_ordering = orderings[:, :, 0]
            orderings = orderings[:, 1:, 1:]
        else:
            atom_ordering = None

        sysnum = len(coordlist)
        coordnum = len(coordlist[0])
        total_points = np.empty((sysnum, coordnum+1, 3))

        # gotta build the first three points by hand but everything else can be constructed iteratively

        # first we put the origin whereever the origins are specified
        if origins is None:
            origins = [0, 0, 0]
        if not isinstance(origins, np.ndarray):
            origins = np.array(origins)
        if len(origins.shape) < 2:
            origins = np.broadcast_to(origins, (sysnum, 3))
        total_points[:, 0] = origins

        # set up the next points by just setting them along the x-axis by default
        if axes is None:
            axes = [1, 0, 0]
        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)
        if axes.ndim == 1:
            axes = np.array([axes, [0, 1, 0]])#np.concatenate((np.random.uniform(low=.5, high=1, size=(2,)), np.zeros((1,)) ))])
        if axes.ndim == 2:
            axes = np.broadcast_to(axes[np.newaxis], (sysnum, 2, 3))
        x_axes = vec_normalize(axes[:, 0])
        dists = coordlist[:, 0, 0]
        ref_points_1 = np.reshape(dists, (sysnum, 1)) * x_axes
        ref_points_1 += origins
        total_points[:, 1] = ref_points_1

         # iteratively build the rest of the coords with one special cases for n=2
        for i in range(1, coordnum):

            # Get the distances away
            # raise Exception(coordlist[:, i, [0, 2, 4]])
            ref_coords1 = orderings[:, i, 0] # reference atom numbers for first coordinate
            refs1 = total_points[np.arange(sysnum), ref_coords1] # get the actual reference coordinates
            dists = np.reshape(coordlist[:, i, 0], (sysnum, 1)) # pull the requisite distances

            ref_coords2 = orderings[:, i, 1] # reference atom numbers for second coordinate
            refs2 = total_points[np.arange(sysnum), ref_coords2] # get the actual reference coordinates for the angle
            angle = coordlist[:, i, 1] # pull the requisite angle values
            if not use_rad:
                angle = np.deg2rad(angle)

            if i == 1:
                refs3 = None
                dihed = None
            else:
                ref_coords3 = orderings[:, i, 2] # reference atom numbers for dihedral ref coordinate
                refs3 = total_points[np.arange(sysnum), ref_coords3] # get the actual reference coordinates for the dihed
                dihed = coordlist[:, i, 2] # pull proper dihedral values
                if not use_rad:
                    dihed = np.deg2rad(dihed)

            ref_points = self.build_next_points(refs1, dists, refs2, angle, refs3, dihed,
                                                ref_axis=axes[:, 1],
                                                return_derivs = return_derivs
                                                ) # iteratively build z-mat
            if return_derivs:
                ref_points, derivs = return_derivs
            total_points[:, i+1] = ref_points

        if atom_ordering is not None:
            rev_ord = np.argsort(atom_ordering, axis=1)
            total_points = total_points[np.arange(len(atom_ordering))[:, np.newaxis], rev_ord]

        return total_points, dict(use_rad=use_rad, ordering=ordering)

    def convert(self, coords, **kw):
        """dipatches to convert_many but only pulls the first"""
        total_points, opts = self.convert_many(np.reshape(coords, (1,)+coords.shape), **kw)[0]
        return total_points[0], opts


__converters__ = [ ZMatrixToCartesianConverter() ]