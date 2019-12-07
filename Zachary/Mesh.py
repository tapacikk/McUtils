"""
Represents an n-dimensional grid, used by Interpolator and (eventually) FiniteDifferenceFunction to automatically
know what kind of data is fed in
"""

import numpy as np

__all__ = [ "Mesh" ]

class Mesh(np.ndarray):
    """
    A general Mesh class representing data points in n-dimensions in either a structured or unstructured manner
    """
    # we define a bunch of MeshType attributes so that the string names can change up or whatever but the interface
    # can remain consistent
    MeshType_Structured = "structured"
    MeshType_Unstructured = "structured"
    MeshType_SemiStructured = "unsemistructured"
    MeshType_Indeterminate = "indeterminate"

    # subclassing np.ndarray is weird... maybe I don't even want to do it... but I _think_ I do
    def __new__(cls,
                data,
                mesh_type = None
                ):
        """
        :param griddata: the raw grid-point data the mesh uses
        :type griddata: np.ndarray
        :param mesh_type: the type of mesh we have
        :type mesh_type: None | str
        :param opts:
        :type opts:
        """
        self = np.asarray(data).view(cls)
        self.mesh_type = self.get_mesh_type(self) if mesh_type is None else mesh_type
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

    @property
    def mesh_spacings(self):
        if self.mesh_type == self.MeshType_Structured:
            return [ m[1] - m[0] for m in self.subgrids ]
        else:
            return None
    @property
    def subgrids(self):
        if self.mesh_type == self.MeshType_Structured:
            unroll = np.roll(np.arange(len(self.shape)), 1)
            meshes = self.transpose(unroll)
            return [np.unique(m) for m in meshes]
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
    @classmethod
    def get_npoints(cls, g):
        """Returns the number of gridpoints in the grid

        :param g:
        :type g: np.ndarray
        :return:
        :rtype: int
        """
        return np.product(g.shape[:-1])
    @property
    def gridpoints(self):
        """Returns the flattened set of gridpoints for a structured tensor grid and otherwise just returns the gridpoints

        :return:
        :rtype:
        """
        return self.get_gridpoints(self)
    @classmethod
    def get_gridpoints(cls, g):
        """Returns the gridpoints in the grid

        :param g:
        :type g: np.ndarray
        :return:
        :rtype: int
        """
        return np.reshape(g, (cls.get_gridpoints(g), g.shape[-1]))

    @classmethod
    def get_mesh_type(cls, grid):
        """Determines what kind of grid we're working with

        :param grid:
        :type grid: np.ndarray
        :return: mesh_type
        :rtype: str
        """
        ndim = grid.ndim
        shape = grid.shape
        if ndim == 1:
            mesh_spacings = np.diff(np.sort(grid))
            if len(np.unique(mesh_spacings)) == 1:
                return cls.MeshType_Structured
            else:
                return cls.MeshType_Unstructured
        elif ndim == 2: # this means we were fed grid points
            grid = np.asarray(grid)
            meshes = grid.T
            mesh_points = [np.unique(x) for x in meshes]
            mesh_lens = [ len(x) for x in mesh_points ]
            points =  np.product(mesh_lens)
            if len(grid) == points:
                # should also check mesh-spacing consistency, but for now we won't
                return cls.MeshType_Structured
            elif len(grid) < points:
                # either semistructured or unstructured
                mesh_spacings = [ len(np.diff(np.sort(g))) for g in mesh_points ]
                if np.all(np.array(mesh_spacings) == 1):
                    return cls.MeshType_SemiStructured
                else:
                    return cls.MeshType_Unstructured
            else:
                # maybe throw an error to complain about duplicate abcissa?
                return cls.MeshType_Indeterminate

        else: # means we _probably_ have a structured grid
            # I should be checking the mesh-spacings but I don't wanna
            if shape[-1] == ndim - 1:
                return cls.MeshType_Structured
            else:
                return cls.MeshType_Indeterminate

    @classmethod
    def RegularMesh(cls, *mesh_specs):
        # should probably handle the empty mesh subcase...
        coords = [ np.linspace(*m) for m in mesh_specs ]
        cg = np.meshgrid(coords)
        roll = np.roll(np.arange(len(mesh_specs)), -1)
        return cls(np.transpose(cg, roll), mesh_type=cls.MeshType_Structured)
