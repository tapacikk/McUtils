"""
Represents an n-dimensional grid, used by Interpolator and (eventually) FiniteDifferenceFunction to automatically
know what kind of data is fed in
"""

import numpy as np, enum

__all__ = [ "Mesh", "MeshType" ]

# we define a bunch of MeshType attributes so that the string names can change up or whatever but the interface
# can remain consistent
class MeshType(enum.Enum):
    Structured = "structured"
    Unstructured = "unstructured"
    SemiStructured = "semistructured"
    Indeterminate = "indeterminate"

class MeshError(Exception):
    ...

class Mesh(np.ndarray):
    """
    A general Mesh class representing data points in n-dimensions
    in either a structured, unstructured, or semi-structured manner.
    Exists mostly to provides a unified interface to difference FD and Surface methods.
    """

    # just to make it accessible through Mesh
    MeshError = MeshError
    MeshType = MeshType


    _allow_indeterminate = False
    # subclassing np.ndarray is weird... maybe I don't even want to do it... but I _think_ I do
    def __new__(cls,
                data,
                mesh_type = None,
                allow_indeterminate = None
                ):
        """
        :param griddata: the raw grid-point data the mesh uses
        :type griddata: np.ndarray
        :param mesh_type: the type of mesh we have
        :type mesh_type: None | str
        :param opts:
        :type opts:
        """
        data = np.asarray(data)
        # data.mesh_type = mesh_type
        aid = cls._allow_indeterminate
        try:
            cls._allow_indeterminate = allow_indeterminate # gotta temp disable this...
            self = data.view(cls)
        finally:
            cls._allow_indeterminate = aid
        self.allow_indeterminate = allow_indeterminate
        self.mesh_type = self.get_mesh_type(self) if mesh_type is None else mesh_type
        self._spacings = None
        return self

    def __init__(self, *args, **kwargs):
        # empty init call because numpy is weird
        ...

    def __array_finalize__(self, mesh):
        # basically just a validator...
        if mesh is None:
            return None

        self.mesh_type = getattr(mesh, "mesh_type", None)
        if self.mesh_type is None:
            self.mesh_type = self.get_mesh_type(self)

        self.allow_indeterminate = getattr(mesh, "allow_indeterminate", None)
        aid = self.allow_indeterminate
        if aid is None:
            aid = self._allow_indeterminate
        if not aid and self.mesh_type is MeshType.Indeterminate:
            raise MeshError("indeterminate MeshType, but `allow_indeterminate` turned off")


    @property
    def mesh_spacings(self):
        if self._spacings is None:
            self._spacings = self.get_mesh_spacings(self)
        return self._spacings
    @property
    def subgrids(self):
        if self.mesh_type == MeshType.Structured:
            return self.get_mesh_subgrids(self)
        else:
            return None
    @property
    def dimension(self):
        """Returns the dimension of the grid (not necessarily ndim)

        :return:
        :rtype: int
        """
        return self.shape[-1]
    @property
    def npoints(self):
        """Returns the number of gridpoints in the mesh

        :return:
        :rtype: int
        """
        return self.get_npoints(self)

    @property
    def gridpoints(self):
        """Returns the flattened set of gridpoints for a structured tensor grid and otherwise just returns the gridpoints

        :return:
        :rtype:
        """
        return self.get_gridpoints(self)

    def get_slice_iter(self, axis=-1):
        """Returns an iterator over the slices of the mesh along the specified axis

        :param axis:
        :type axis:
        :return:
        :rtype:
        """
        import itertools as ip

        if isinstance(axis, (int, np.integer)):
            axis = [axis]

        raise NotImplementedError("Need to finish this up")

        ndim = self.dimension
        sdim = len(axis)
        axis = [ ndim+a if a < 0 else a for a in axis ]
        if self.mesh_type == MeshType.Structured:
            grid = np.asarray(self)
            t_spec = [ i for i in range(ndim) if i not in axis ] + axis
            grid = grid.transpose(t_spec)
            for ind in ip.product(grid.shape[-sdim:]):
                yield grid[(...,) + ind]
        elif self.mesh_type == MeshType.SemiStructured:
            # means ndim == 2
            grid = np.asarray(self)
            meshes = grid.T


        else:
            raise MeshError("{}.{}: can't get slices for mesh type {}".format(
                type(self).__name__,
                "get_slice_iter",
                self.mesh_type
            ))

    @classmethod
    def get_npoints(cls, g):
        """Returns the number of gridpoints in the grid

        :param g:
        :type g: np.ndarray
        :return:
        :rtype: int
        """
        return np.product(g.shape[:-1])
    @classmethod
    def get_gridpoints(cls, g):
        """Returns the gridpoints in the grid

        :param g:
        :type g: np.ndarray
        :return:
        :rtype: int
        """
        return np.reshape(g, (cls.get_npoints(g), g.shape[-1]))

    @classmethod
    def get_mesh_subgrids(cls, grid, tol=8):
        if grid.ndim == 1:
            return [ grid ]
        elif grid.ndim ==2:
            if isinstance(grid, Mesh):
                grid = np.array(grid)
            meshes = grid.T
            mesh_points = [np.unique(np.round(x, tol)) for x in meshes]
            return mesh_points
        else:
            unroll = np.roll(np.arange(len(grid.shape)), 1)
            meshes = np.asarray(grid).transpose(unroll)
            return [np.unique(np.round(m, tol)) for m in meshes]

    @classmethod
    def get_mesh_spacings(cls, grid, tol=8):
        ndim = grid.ndim
        if ndim == 1:
            grid = np.asarray(grid)
            mesh_spacings = np.round(np.diff(np.sort(grid)), tol)
        else:
            subgrids = cls.get_mesh_subgrids(grid, tol=tol)
            mesh_spacings = [np.unique(np.diff(np.sort(g))) for g in subgrids]

        return mesh_spacings

    @staticmethod
    def _is_meshgrid(grid):
        """
        Just checks if the shape of the grid
        is consistent with being a meshgrid
        :param grid:
        :type grid:
        :return:
        :rtype:
        """

        shape = grid.shape
        return (
            len(shape) > 1
            and any(shape[0] != s for s in shape[1:])
            and (shape[0] == len(shape) - 1)
        ) # might introduce edge-case...? Hard to know

    @classmethod
    def get_mesh_type(cls, grid, tol=8):
        """Determines what kind of grid we're working with

        :param grid:
        :type grid: np.ndarray
        :return: mesh_type
        :rtype: MeshType
        """

        grid = np.asarray(grid)

        if grid.dtype == np.dtype(object):
            return MeshType.Indeterminate

        ndim = grid.ndim
        if cls._is_meshgrid(grid):
            roll = np.roll(np.arange(ndim), -1)
            grid = grid.transpose(roll)
        shape = grid.shape

        mesh_spacings = cls.get_mesh_spacings(grid, tol=tol)
        if ndim == 1:
            if mesh_spacings[0] is not None:
                return MeshType.Structured
            else:
                return MeshType.Unstructured
        elif ndim == 2: # this likely means we were fed grid points
            subgrids = cls.get_mesh_subgrids(grid, tol=tol)
            mesh_lens = [len(x) for x in subgrids]
            points = np.product(mesh_lens)
            consistent_spacing = all(x is not None for x in mesh_spacings)
            if len(grid) == points and consistent_spacing:
                # should also check mesh-spacing consistency, but for now we won't
                return MeshType.Structured
            elif len(grid) < points:
                # either semistructured or unstructured
                if consistent_spacing:
                    return MeshType.SemiStructured
                else:
                    return MeshType.Unstructured
            else:
                # maybe throw an error to complain about duplicate abcissae?
                return MeshType.Indeterminate
        else: # means we _probably_ have a structured grid
            consistent_spacing = all(x is not None for x in mesh_spacings)
            if shape[-1] == ndim - 1 and consistent_spacing:
                return MeshType.Structured
            else:
                return MeshType.Indeterminate

    @classmethod
    def RegularMesh(cls, *mesh_specs):
        """
        Builds a grid from multiple linspace arguments,
        basically insuring it's structured (if non-Empty)
        :param mesh_specs:
        :type mesh_specs:
        :return:
        :rtype:
        """
        # should probably handle the empty mesh subcase...
        coords = [ np.linspace(*m) for m in mesh_specs ]

        if any(len(x) == 0 for x in coords):
            raise ValueError()
        cg = np.meshgrid(*coords)
        roll = np.roll(np.arange(len(coords) + 1), -1)
        wat = np.transpose(cg, roll)
        return cls(wat, mesh_type=MeshType.Structured)
