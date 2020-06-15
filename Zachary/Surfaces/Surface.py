
from .BaseSurface import *
import numpy as np

__all__ = [
    "Surface",
    "MultiSurface",
    "DipoleSurface"
]

class Surface:
    """
    This actually isn't a concrete implementation of BaseSurface.
    Instead it's a class that _dispatches_ to an implementation of BaseSurface to do its core evaluations (plus it does shape checking)
    """
    def __init__(self, data, dimension=None, base=None, **metadata):
        """

        :param data:
        :type data:
        :param dimension:
        :type dimension:
        :param base:
        :type base: None | Type[BaseSurface]
        :param metadata:
        :type metadata:
        """
        self.metadata = metadata

        if len(data) == 2 and isinstance(data[1], dict):
            data, opts = data
        elif isinstance(data, dict):
            opts = data
            data = ()
        else:
            opts = {}
        if base is None:
            base = self.detect_base(data, opts)
        self.base = base(*data, dimension=dimension, **opts)

    @classmethod
    def detect_base(cls, data, opts):
        """
        Infers what type of base surface works for the data that's passed in.
        It's _super_ roughly done so...yeah generally better to pass the base class you want explicitly.
        But in the absence of that we can do this ?_?

        Basic strategy:
            1. look for options that go with specific methods
            2. look at data structures to guess
                i.   gradient as the first data arg + all data args are ndarrays -> Taylor Series
                ii.  callables as second arg -> Linear expansion or Linear fit
                iii. just like...one big array -> Interpolatin

        :param data:
        :type data: tuple
        :param opts:
        :type opts: dict
        :return:
        :rtype:
        """
        from ..FittableModel import LinearFitBasis

        # try a Taylor expansion
        if (
                any(key in opts for key in ['center', 'ref', 'weight_coefficients']) or (
                all(isinstance(d, np.ndarray) for d in data) and data[0].ndim==1
            )
        ):
            return TaylorSeriesSurface
        # next look into fitting
        elif (
                any(key in opts for key in ['basis']) or (
                len(data)==2 and (isinstance(data[1], LinearFitBasis) or all(callable(d) for d in data[1]))
            )
        ):
            # See if we were given fit coefficients or actual data to fit
            if data[0].ndim == 1:
                return LinearExpansionSurface
            else:
                return LinearFitSurface
        else:
            return InterpolatedSurface

    def __call__(self, gridpoints, **kwargs):
        return self.base(gridpoints, **kwargs)

class MultiSurface:
    """
    A _reallly_ simple extension to the Surface infrastructure to handle vector valued functions,
    assuming each vector value corresponds to a different Surfaces
    """
    def __init__(self, *surfs):
        """

        :param surfs: a set of Surface objects to use when evaluating
        :type surfs: Iterable[Surface]
        """
        self.surfs = surfs
    def __call__(self, gridpoints, **kwargs):
        return np.column_stack([s(gridpoints, **kwargs) for s in self.surfs])

class DipoleSurface(MultiSurface):
    """
    WARNING TO FUTURE PEOPLE: this is almost certainly going to move to a separate package,
    once I figure out what that package should be...
    """
    def __init__(self, mu_x, mu_y, mu_z):
        """

        :param mu_x: X-component of dipole moment
        :type mu_x: Surface
        :param mu_y: Y-component of dipole moment
        :type mu_y: Surface
        :param mu_z: Z-component of dipole moment
        :type mu_z: Surface
        """
        if isinstance(mu_x.base, TaylorSeriesSurface):
            self.mode = "taylor"
        else:
            self.mode = "interp"
        super().__init__(
            mu_x,
            mu_y,
            mu_z
        )
    @classmethod
    def from_log_file(cls, log_file, coord_transf, tol = .001, keys = ("StandardCartesianCoordinates", "DipoleMoments"), **opts):
        """
        Loads dipoles from a Gaussian log file and builds a dipole surface by interpolating.
        Obviously this only really works if we have a subset of "scan" coordinates, so at this stage the user is obligated
        to furnish a function that'll take a set of Cartesian coordinates and convert them to "scan" coordinates.
        Coordinerds can be helpful with this, as it provides a convenient syntax for Cartesian <-> ZMatrix conversions

        :param log_file: a Gaussian log file to pull from
        :type log_file: str
        :return:
        :rtype:
        """

        from ...GaussianInterface import GaussianLogReader

        with GaussianLogReader(log_file) as parser:
            parse_data = parser.parse(keys)

        carts = parse_data[keys[0]][1]
        dipoles = parse_data[keys[1]]
        scan_coords = coord_transf(carts)
        if len(dipoles) != len(scan_coords):
            raise ValueError(
                "mismatch between number of dipoles ({}) and number of coordinates ({})".format(
                    len(dipoles),
                    len(scan_coords)
                )
            )

        if scan_coords.ndim == 1:
            scan_sort = np.argsort(scan_coords)
        else:
            scan_sort = np.lexsort(tuple(reversed(tuple(scan_coords.T))))
        scan_coords = scan_coords[scan_sort]
        dipoles = dipoles[scan_sort]

        # this is particularly relevant for optimization scans...but we pull all the "unique" indices
        # then we pull the indices right _before_ each unique one since that's the final one in the block of "uniques"
        # finally we do a "roll" to make sure the order from the sort is preserved
        tol_coords = np.floor(scan_coords/tol)
        if tol_coords.ndim == 1:
            diffs = np.diff(tol_coords)
        else:
            print(tol_coords)
            diffs = np.sum(abs(np.diff(tol_coords, axis=0)), axis=1)
        inds = np.where(diffs != 0)[0]
        inds = np.concatenate((inds, [len(inds)]))
        scan_coords = scan_coords[inds]
        dipoles = dipoles[inds]

        dipoles = list(np.transpose(dipoles))

        return MultiSurface(*(
            Surface(
                ((scan_coords, d), opts),
                base = InterpolatedSurface,
                dipole_component = "x" if i == 0 else "y" if i == 1 else "z"
            ) for i,d in enumerate(dipoles)
        ))

    @classmethod
    def from_fchk_file(cls, fchk_file, **opts):
        """
        Loads dipoles from a Gaussian formatted checkpoint file and builds a dipole surface via a linear approximation

        :param fchk_file: a Gaussian fchk file to pull from
        :type log_file: str
        :return:
        :rtype:
        """

        from ...GaussianInterface import GaussianFChkReader

        with GaussianFChkReader(fchk_file) as parser:
            parse_data = parser.parse(["Coordinates", "Dipole Moment", "Dipole Derivatives"])

        center = parse_data["Coordinates"]
        const_dipole = parse_data["Dipole Moment"]
        derivs = parse_data["Dipole Derivatives"]
        derivs = np.reshape(derivs, (int(len(derivs)/3), 3))
        derivs = list(np.transpose(derivs))

        opts['center'] = center.flatten()
        surfs = [None]*3
        for i, d in enumerate(zip(derivs, list(const_dipole))):
            d, r = d
            opts = opts.copy()
            opts["ref"] = r
            surfs[i] = Surface(
                ((d,), opts),
                base = TaylorSeriesSurface,
                dipole_component="x" if i == 0 else "y" if i == 1 else "z"
            )

        return cls(*surfs)

    def __call__(self, gridpoints, **opts):
        """
        Explicitly overrides the Surface-level evaluation because we know the Taylor surface needs us to flatten our gridpoints

        :param gridpoints:
        :type gridpoints:
        :param opts:
        :type opts:
        :return:
        :rtype:
        """

        gps = np.asarray(gridpoints)
        if self.mode == "taylor":
            if gps.ndim == 2:
                gps = gps.flatten()
            elif gps.ndim > 2:
                gps = np.reshape(gps, gps.shape[:-2] + (np.product(gps.shape[-2:]),))

        return super().__call__(gps, **opts)

