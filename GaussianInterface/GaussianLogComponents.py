"""This lists the types of readers and things available to the GaussianLogReader

"""

from ..Parsers import *

########################################################################################################################
#
#                                           GaussianLogComponents
#
# region GaussianLogComponents
GaussianLogComponents = {}  # we'll register on this bit by bit
# each registration should look like:

# GaussianLogComponents["Name"] = {
#     "description" : string, # used for docmenting what we have
#     "tag_start"   : start_tag, # starting delimeter for a block
#     "tag_end"     : end_tag, # ending delimiter for a block None means apply the parser upon tag_start
#     "parser"      : parser, # function that'll parse the returned list of blocks (for "List") or block (for "Single")
#     "mode"        : mode # "List" or "Single"
# }

########################################################################################################################
#
#                                           InputZMatrix
#

# region InputZMatrix
tag_start = "Z-matrix:"
tag_end   = """ 
"""

def parser(zmat):
    return zmat

mode = "Single"

GaussianLogComponents["InputZMatrix"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           CartesianCoordinates
#

# region CartesianCoordinates

 # the region thing is just a PyCharm hack to collapse the boilerplate here... Could also have done 5000 files

cart_delim = """ --------------------------------------------------------------"""
cartesian_start_tag = FileStreamerTag(
    """Center     Atomic      Atomic             Coordinates (Angstroms)""",
    follow_ups= cart_delim
)
cartesian_end_tag = cart_delim

CartParser = StringParser(
    Repeating(
        (
            Named(
                Repeating(Capturing(PositiveInteger), min=3, max=3, prefix=Optional(Whitespace), suffix=Whitespace),
                "GaussianStuff"
            ),
            Named(
                Repeating(
                    Capturing(Number),
                    min = 3,
                    max = 3,
                    prefix=Optional(Whitespace),
                    joiner = Whitespace
                ),
                "Coordinates"
            )
        ),
        suffix = Optional(Newline)
    )
)

def cartesian_coordinates_parser(strs):
    strss = "\n\n".join(strs)

    parse = CartParser.parse_all(strss)

    coords = (
        parse["GaussianStuff", 0],
        parse["Coordinates"].array
    )

    return coords

GaussianLogComponents["CartesianCoordinates"] = {
    "tag_start": cartesian_start_tag,
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["ZMatCartesianCoordinates"] = {
    "tag_start": FileStreamerTag('''Z-Matrix orientation:''', follow_ups = (cart_delim, cart_delim)),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["StandardCartesianCoordinates"] = {
    "tag_start": FileStreamerTag('''Standard orientation:''', follow_ups = (cart_delim, cart_delim)),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["InputCartesianCoordinates"] = {
    "tag_start": FileStreamerTag('''Input orientation:''', follow_ups = (cart_delim, cart_delim)),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}

# endregion

########################################################################################################################
#
#                                           ZMatrices
#

# region ZMatrices
tag_start = """Z-MATRIX (ANGSTROMS AND DEGREES)
   CD    Cent   Atom    N1       Length/X        N2       Alpha/Y        N3        Beta/Z          J
 ---------------------------------------------------------------------------------------------------"""
tag_end   = " ---------------------------------------------------------------------"

ZMatParser = StringParser(
    Repeating(
        (
            Named(
                Repeating(Capturing(PositiveInteger), min=1, max=2, prefix=Optional(Whitespace), suffix=Whitespace),
                "GaussianInts"
            ),
            Named(
                Capturing(AtomName),
                "AtomNames",
                suffix=Whitespace
            ),
            Named(
                Repeating(
                    (
                        Capturing(PositiveInteger),
                        Capturing(Number),
                        Parenthesized(PositiveInteger, prefix=Whitespace)
                    ),
                    min = None,
                    max = 3,
                    prefix=Optional(Whitespace),
                    joiner = Whitespace
                ),
                "Coordinates"
            )
        ),
        suffix = Optional(RegexPattern((
            Optional((Whitespace, PositiveInteger)),
            Optional(Newline)
        )))
    )
)

def parser(strs):

    strss = '\n\n'.join(strs)
    fak = ZMatParser.parse_all(strss)
    coords = [
        (
            fak["GaussianInts", 0],
            fak["AtomNames", 0]
        ),
        fak["Coordinates", 0, 0],
        fak["Coordinates", 1]
        ]

    return coords
mode = "List"

GaussianLogComponents["ZMatrices"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           OptimizationParameters
#

# region OptimizationParameters

tag_start  = "Optimization "
tag_end    = """                        !
 ------------------------------------------------------------------------
"""


def parser(pars):
    """Parses a optimizatioon parameters block"""
    did_opts = [ "Non-Optimized" not in par for par in pars]
    return did_opts, pars


mode = "List"

GaussianLogComponents["OptimizationParameters"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           MullikenCharges
#

#region MullikenCharges
tag_start = "Mulliken charges:"
tag_end   = "Sum of Mulliken charges"


def parser(charges):
    """Parses a Mulliken charges block"""
    return charges
mode = "List"

GaussianLogComponents["MullikenCharges"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           MultipoleMoments
#

#region MultipoleMoments
tag_start  = "Dipole moment ("
tag_end    = " N-N="


def parser(moms):
    """Parses a multipole moments block"""
    return moms


mode = "List"

GaussianLogComponents["MultipoleMoments"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           DipoleMoments
#

# region DipoleMoments
tag_start  = "Dipole moment ("
tag_end    = "Quadrupole moment ("

dips_parser = StringParser(
    RegexPattern(
        (
            "X=", Capturing(Number),
            "Y=", Capturing(Number),
            "Z=", Capturing(Number)
        ),
        joiner=Whitespace
    )
)
def parser(moms):
    """Parses a multipole moments block"""
    # print(repr(str(dips_parser.regex)), file=sys.stderr)
    res = dips_parser.parse_all("\n".join(moms))
    return res.array
mode       = "List"


mode = "List"
GaussianLogComponents["DipoleMoments"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"    : parser,
    "mode"      : mode
}

# endregion

########################################################################################################################
#
#                                           OptimizedDipoleMoments
#

# region DipoleMoments
tag_start  = " Dipole        ="
tag_end    = " Optimization"

DNumberPattern = RegexPattern((Number, "D", PositiveInteger), dtype = str)
OptimizedDipolesParser = StringParser(
    RegexPattern(
        (
            "Dipole", "=",
            Repeating(
                Capturing(DNumberPattern),
                min=3,
                max=3,
                suffix=Optional(Whitespace)
            )
        ),
        joiner=Whitespace
    )
)

def parser(mom):
    """Parses dipole block, but only saves the dipole of the optimized structure"""
    import numpy as np

    mom = "Dipole  =" + mom
    grps = OptimizedDipolesParser.parse_all(mom)
    grp = grps[-1]
    dip_list = [x.replace("D", "E") for x in grp]
    dip_array = np.asarray(dip_list)
    return dip_array.astype("float64")
mode       = "List"
parse_mode = "Single"

GaussianLogComponents["OptimizedDipoleMoments"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"    : parser,
    "mode"      : mode,
    "parse_mode": parse_mode
}

# endregion

########################################################################################################################
#
#                                           ScanEnergies
#

# region ScanEnergies

tag_start = """ Summary of the potential surface scan:"""
tag_end = """ Leave Link  108"""

# Number = '(?:[\\+\\-])?\\d*\\.\\d+'
# block_pattern = "\s*"+Number+"\s*"+Number+"\s*"+Number+"\s*"+Number+"\s*"+Number
# block_re = re.compile(block_pattern)

def parser(pars):
    """Parses the scan summary block"""
    import numpy as np

    vals = np.array(pars, dtype = float)
    return vals

mode = "Single"

GaussianLogComponents["ScanEnergies"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

########################################################################################################################
#
#                                           OptimizedScanEnergies
#

# region OptimizedScanEnergies

tag_start = """ Summary of Optimized Potential Surface Scan"""
tag_end = FileStreamerTag(
    tag_alternatives = (
        """ Largest change from initial coordinates is atom """,
        """-"""*25
    )
)


eigsPattern = RegexPattern(
    (
        "Eigenvalues --",
        Repeating(Capturing(Number), suffix=Optional(Whitespace))
    ),
    joiner=Whitespace
)

coordsPattern = RegexPattern(
    (
        Capturing(VariableName),
        Repeating(Capturing(Number), suffix=Optional(Whitespace))
    ),
    prefix=Whitespace,
    joiner=Whitespace
)

OptScanPat = StringParser(
    RegexPattern(
        (
            Named(eigsPattern,
                  "Eigenvalues"
                  #parser=lambda t: np.array(Number.findall(t), 'float')
                  ),
            Named(Repeating(coordsPattern, suffix=Optional(Newline)), "Coordinates")
        ),
        joiner=Newline
    )
)

def parser(pars):
    """Parses the scan summary block and returns only the energies"""
    from collections import OrderedDict
    import numpy as np

    if pars is None:
        return None

    par_data = OptScanPat.parse_all(pars)
    energies_array = np.concatenate(par_data["Eigenvalues"])
    coords = OrderedDict()
    cdz = [a.array for a in par_data["Coordinates"].array]
    for coord_names, coord_values in zip(*cdz):
        for k, v in zip(coord_names, coord_values):
            if k not in coords:
                coords[k] = [v]
            else:
                coords[k].append(v)
    for k in coords:
        coords[k] = np.concatenate(coords[k])

    return energies_array, coords


mode = "Single"

GaussianLogComponents["OptimizedScanEnergies"] = {
    "tag_start": tag_start,
    "tag_end"  : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

# endregion

# endregion

########################################################################################################################
#
#                                           GaussianLogDefaults
#
# region GaussianLogDefaults
GaussianLogDefaults = (
    "StartDateTime",
    "InputZMatrix",
    "ScanTable",
    "Blurb",
    "ComputerTimeElapsed",
    "EndDateTime"
)
# endregion

########################################################################################################################
#
#                                           GaussianLogOrdering
#
# region GaussianLogOrdering
# defines the ordering in a GaussianLog file
glk = ( # this must be sorted by what appears when
    "StartDateTime",
    "AtomPositions",
    "CartesianCoordinates",
    "CartesianCoordinateVectors",
    "MullikenCharges",
    "MultipoleMoments",
    "DipoleMoments",
    "QuadrupoleMoments",
    "OctapoleMoments",
    "HexadecapoleMoments",
    "HartreeFockEnergies",
    "MP2Energies",
    "InputZMatrix",
    "InputZMatrixVariables",
    "ZMatrices",
    "ZMatrixCoordinates",
    "ZMatrixCoordinateVectors",
    "ScanTable",
    "OptimizationScan",
    "Blurb",
    "ComputerTimeElapsed",
    "EndDateTime"
)
GaussianLogOrdering = { k:i for i, k in enumerate(glk) }
del glk
# endregion
