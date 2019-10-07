"""This lists the types of readers and things available to the GaussianLogReader

"""

from ..Parsers import *

########################################################################################################################
#
#                                           GaussianLogComponents
#
#region GaussianLogComponents
GaussianLogComponents = { } # we'll register on this bit by bit
# each registration should look like:

# GaussianLogComponents["Name"] = {
#     "description" : string, # used for docmenting what we have
#     "tag_start"    : start_tag, # starting delimeter for a block
#     "tag_end"      : end_tag, # ending delimiter for a block None means apply the parser upon tag_start
#     "parser"      : parser, # function that'll parse the returned list of blocks (for "List") or single block (for "Single)
#     "mode"        : mode # "List" or "Single"
#
# }

########################################################################################################################
#
#                                           InputZMatrix
#

#region InputZMatrix
tag_start  = "Z-matrix:"
tag_end    = """ 
"""
def parser(zmat):
    return zmat
mode       = "Single"

GaussianLogComponents["InputZMatrix"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

#endregion

########################################################################################################################
#
#                                           CartesianCoordinates
#

#region CartesianCoordinates

 # the region thing is just a PyCharm hack to collapse the boilerplate here... Could also have done 5000 files

cart_delim = """ --------------------------------------------------------------"""
plain_cartesian_start_tags = (
    """Center     Atomic      Atomic             Coordinates (Angstroms)""",
    cart_delim
    )
cartesian_end_tag = cart_delim

# I'm gonna have to update this to work with the expanded Regex support...
# cartesian_re_c = re.compile(ws_p.join(["("+int_p+")"]*3)+ws_p+cart_p)

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
    "tag_start" : plain_cartesian_start_tags,
    "tag_end"   : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["ZMatCartesianCoordinates"] = {
    "tag_start" : ('''Z-Matrix orientation:''', cart_delim, cart_delim),
    "tag_end"   : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["StandardCartesianCoordinates"] = {
    "tag_start" : ('''Standard orientation:''', cart_delim, cart_delim),
    "tag_end"   : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["InputCartesianCoordinates"] = {
    "tag_start" : ('''Input orientation:''', cart_delim, cart_delim),
    "tag_end"   : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}

#endregion

########################################################################################################################
#
#                                           ZMatrices
#

#region ZMatrices
tag_start  = """Z-MATRIX (ANGSTROMS AND DEGREES)
   CD    Cent   Atom    N1       Length/X        N2       Alpha/Y        N3        Beta/Z          J
 ---------------------------------------------------------------------------------------------------"""
tag_end    = " ---------------------------------------------------------------------"

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
mode       = "List"

GaussianLogComponents["ZMatrices"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

#endregion

########################################################################################################################
#
#                                           OptimizationParameters
#

#region OptimizationParameters

tag_start  = "Optimization "
tag_end    = """                        !
 ------------------------------------------------------------------------
"""
def parser(pars):
    """Parses a optimizatioon parameters block"""
    did_opts = [ "Non-Optimized" not in par for par in pars]
    return did_opts, pars
mode       = "List"

GaussianLogComponents["OptimizationParameters"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

#endregion

########################################################################################################################
#
#                                           MullikenCharges
#

#region MullikenCharges
tag_start  = "Mulliken charges:"
tag_end    = "Sum of Mulliken charges"
def parser(charges):
    """Parses a Mulliken charges block"""
    return charges
mode       = "List"

GaussianLogComponents["MullikenCharges"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

#endregion

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
mode       = "List"

GaussianLogComponents["MultipoleMoments"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"   : parser,
    "mode"     : mode
}

#endregion

########################################################################################################################
#
#                                           DipoleMoments
#

#region DipoleMoments
tag_start  = "Dipole moment ("
tag_end    = "Quadrupole moment ("

# get_dips_pat = "X=\s+"+grp_p(num_p)+"\s+Y=\s+"+grp_p(num_p)+"\s+Z=\s+"+grp_p(num_p)
# get_dips_re = re.compile(get_dips_pat)

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

GaussianLogComponents["DipoleMoments"] = {
    "tag_start" : tag_start,
    "tag_end"   : tag_end,
    "parser"    : parser,
    "mode"      : mode
}

#endregion

########################################################################################################################
#
#                                           OptimizedDipoleMoments
#

#region DipoleMoments
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

#endregion

#endregion

########################################################################################################################
#
#                                           GaussianLogDefaults
#
#region GaussianLogDefaults
GaussianLogDefaults = (
    "StartDateTime",
    "InputZMatrix",
    "ScanTable",
    "Blurb",
    "ComputerTimeElapsed",
    "EndDateTime"
)
#endregion

########################################################################################################################
#
#                                           GaussianLogOrdering
#
#region GaussianLogOrdering
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
#endregion