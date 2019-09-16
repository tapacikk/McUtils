"""This lists the types of readers and things available to the GaussianLogReader

"""

from ..Parsers.ParserUtils import *

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
plain_cartesian_start_tags = (
    """Center     Atomic      Atomic             Coordinates (Angstroms)""",
    cart_delim
    )
cartesian_end_tag = cart_delim

cartesian_re_c = re.compile(ws_p.join(["("+int_p+")"]*3)+ws_p+cart_p)


def cartesian_coordinates_parser(strs):
    num_sets = len(strs)
    strit = iter(strs)
    if num_sets>0:
        xyz = next(strit)
        first = pull_xyz(xyz, regex=cartesian_re_c)
        num_atoms = len(first[0])
        if num_sets>1:
            coord_array = np.zeros((num_sets, num_atoms, 3), dtype=np.float64)
            coord_array[0] = first[1]
            for i, xyz in enumerate(strs):
                if i>0:
                    coord_array[i] = pull_coords(xyz)
            coords = [first[0], coord_array]
        else:
            coords = [first[0], np.reshape(first[1], (1,) + first[1].shape)]
    else:
        coords = None

    return coords


GaussianLogComponents["CartesianCoordinates"] = {
    "tag_start": plain_cartesian_start_tags,
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["ZMatCartesianCoordinates"] = {
    "tag_start": ('''Z-Matrix orientation:''', cart_delim, cart_delim),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["StandardCartesianCoordinates"] = {
    "tag_start": ('''Standard orientation:''', cart_delim, cart_delim),
    "tag_end"  : cartesian_end_tag,
    "parser"   : cartesian_coordinates_parser,
    "mode"     : "List"
}
GaussianLogComponents["InputCartesianCoordinates"] = {
    "tag_start": ('''Input orientation:''', cart_delim, cart_delim),
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

gaussian_zzz = op_p(posint_p) + opnb_p(wsr_p) + \
                op_p(posint_p) + opnb_p(wsr_p) + \
                grp_p(name_p+opnb_p(paren_p))
for i in range(3):
    gaussian_zzz += opnb_p( # optional non-binding
        wsr_p + grp_p(posint_p) + # ref int
        wsr_p + grp_p(num_p) + # z-mat value
        opnb_p(paren_p) # ignore parens that Gaussian puts at the end
    )
# print(gaussian_zzz)
gaussian_zzz_c = re.compile(gaussian_zzz)


def parser(strs):
    num_sets = len(strs)
    strit = iter(strs)
    if num_sets>0:
        zm = next(strit)
        # print(zm)
        first = pull_zmat(zm, regex=gaussian_zzz_c, num_header=3)
        # print(first)
        num_atoms = len(first[0])
        if num_sets>1:
            index_array = first[1]
            coord_array = np.zeros((num_sets, num_atoms-1, 3), dtype=np.float64)
            coord_array[0] = first[2]
            for i, zm in enumerate(strs):
                if i>0:
                    coord_array[i] = pull_zmat_coords(zm)
            coords = [first[0], index_array, coord_array]
        else:
            coords = [first[0], first[1], np.reshape(first[2], (1,) + first[2].shape)]
    else:
        coords = None

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

get_dips_pat = "X=\s+"+grp_p(num_p)+"\s+Y=\s+"+grp_p(num_p)+"\s+Z=\s+"+grp_p(num_p)
get_dips_re = re.compile(get_dips_pat)


def parser(moms):
    """Parses a multipole moments block"""
    dippz = [ None ]*len(moms)
    for i, dip in enumerate(moms):
        grps = re.findall(get_dips_re, dip)[0]
        dippz[i] = grps
    dip_list = np.array(dippz, dtype=str)
    return dip_list.astype("float64")


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

dnum_p = num_p + "D" + int_p
get_optdips_pat = "Dipole\s+="+"\s*"+grp_p(dnum_p)+"\s*"+grp_p(dnum_p)+"\s*"+grp_p(dnum_p)
get_optdips_re = re.compile(get_optdips_pat)


def parser(mom):
    """Parses dipole block, but only saves the dipole of the optimized structure"""
    mom = "Dipole  =" + mom
    grps = re.findall(get_optdips_re, mom)
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
    vals = np.array(pars)
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
tag_end = """ Largest change from initial coordinates is atom """


def parser(pars):
    """Parses the scan summary block and returns only the energies"""
    import re
    from collections import OrderedDict

    # eigsPatternShit = RegexPattern(
    #     ("Eigenvalues --", Capturing( Repeating(Number, suffix=Optional(Whitespace)) )),
    #     joiner=Whitespace
    # )

    eigsPatternShit = '(?:Eigenvalues --)(?:(?!\\n)\\s)*((?:(?:[\\+\\-])?\\d*\\.\\d+(?:(?:(?!\\n)\\s)*)?)+)'

    # coordsPatternShit = RegexPattern(
    #     (
    #         Capturing((ASCIILetter, Word), joiner=""),
    #         Capturing(
    #             Repeating(Number, suffix=Optional(Whitespace))
    #         )
    #     ),
    #     prefix=Whitespace,
    #     joiner=Whitespace
    # )

    coordsPatternShit = '(?:(?!\\n)\\s)*([a-zA-Z]\\w+)(?:(?!\\n)\\s)*((?:(?:[\\+\\-])?\\d*\\.\\d+(?:(?:(?!\\n)\\s)*)?)+)'

    # full_pattern = Capturing(
    #     (
    #         eigsPatternShit,
    #         Repeating(coordsPatternShit, suffix=Optional(Newline))
    #     ),
    #     joiner=Newline
    # )

    full_pattern = '((?:Eigenvalues --)(?:(?!\\n)\\s)*((?:(?:[\\+\\-])?\\d*\\.\\d+(?:(?:(?!\\n)\\s)*)?)+)\n(?:(?:(?!\\n)\\s)*([a-zA-Z]\\w+)(?:(?!\\n)\\s)*((?:(?:[\\+\\-])?\\d*\\.\\d+(?:(?:(?!\\n)\\s)*)?)+)(?:\n)?)+)'

    # print(repr(str(Number)))

    Number = '(?:[\\+\\-])?\\d*\\.\\d+'

    numPattern = re.compile(Number)
    eigsPattern = re.compile(eigsPatternShit)
    full_patternPattern = re.compile(full_pattern)
    coordsPatternShitPattern = re.compile(coordsPatternShit)

    energies_array = []
    coords = OrderedDict()

    for match in re.finditer(full_patternPattern, pars):
        block_text = match.groups(0)[0]
        # pull the energies from a block
        energies = re.search(eigsPattern, block_text).groups(0)[0]
        energies = re.findall(numPattern, energies)
        energies = np.array(energies, dtype=float)
        energies_array.append(energies)

        for coord_match in re.finditer(coordsPatternShitPattern, block_text):
            name, coord = coord_match.groups()
            coord = re.findall(numPattern, coord)
            coord = np.array(coord, dtype=float)
            if name not in coords:
                coords[name] = []
            coords[name].append(coord)

    energies_array = np.concatenate(energies_array)
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
