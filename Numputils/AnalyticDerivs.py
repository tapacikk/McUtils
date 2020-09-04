"""
Provides analytic derivatives for some common base terms with the hope that we can reuse them elsewhere
"""
import numpy as np
from .VectorOps import *

__all__ = [
    'levi_cevita3',
    'rot_deriv',
    'dist_deriv',
    'angle_deriv',
    'dihed_deriv'
]

# felt too lazy to look up some elegant formula
levi_cevita3 = np.array([
    [[0, 0, 0], [ 0,0,1],[0,-1,0]],
    [[0, 0,-1], [ 0,0,0],[1, 0,0]],
    [[0, 1, 0], [-1,0,0],[0, 0,0]]
])
# levi_cevita3.__name__ = "levi_cevita3"
# levi_cevita3.__doc__ = """
#     The 3D Levi-Cevita tensor.
#     Used to turn cross products into matmuls
#     """

def rot_deriv(angle, axis, dAngle, dAxis):
    """
    Gives a rotational derivative w/r/t some unspecified coordinate
    (you have to supply the chain rule terms)

    :param angle: angle for rotation
    :type angle: float
    :param axis: axis for rotation
    :type axis: np.ndarray
    :param dAngle: chain rule angle deriv.
    :type dAngle: float
    :param dAxis: chain rule axis deriv.
    :type dAxis: np.ndarray
    :return: derivatives of the rotation matrices with respect to both the angle and the axis
    :rtype: np.ndarray
    """

    # Will still need work to be appropriately vectorized (can't remember if I did this or not?)

    vdOdv = vec_outer(dAxis, axis) + vec_outer(axis, dAxis)
    c = np.cos(angle)
    s = np.sin(angle)
    e3 = levi_cevita3
    i3 = np.eye(3)
    ct = vdOdv*(1-np.cos(angle))
    st = (i3-vec_outer(axis, axis))*np.sin(angle)*dAngle
    et = np.dot(e3, (dAxis*s + axis*c*dAngle))
    return ct - st - et

def dist_deriv(coords, i, j):
    """
    Gives the derivative of the distance between i and j with respect to coords i and coords j

    :param coords:
    :type coords: np.ndarray
    :param i: index of one of the atoms
    :type i: int | Iterable[int]
    :param j: index of the other atom
    :type j: int | Iterable[int]
    :return: derivatives of the distance with respect to atoms i, j, and k
    :rtype: np.ndarray
    """
    v = vec_normalize(coords[j]-coords[i])

    return np.array([-v, v])

def angle_deriv(coords, i, j, k):
    """
    Gives the derivative of the angle between i, j, and k with respect to the Cartesians

    :param coords:
    :type coords: np.ndarray
    :param i: index of the central atom
    :type i: int | Iterable[int]
    :param j: index of one of the outside atoms
    :type j: int | Iterable[int]
    :param k: index of the other outside atom
    :type k: int | Iterable[int]
    :return: derivatives of the angle with respect to atoms i, j, and k
    :rtype: np.ndarray
    """

    dot = vec_dots
    tdo = vec_tdot
    a = coords[j] - coords[i]
    b = coords[k] - coords[i]
    e3 = np.broadcast_to(levi_cevita3, (len(a), 3, 3, 3))
    axb = vec_crosses(a, b)
    adb = vec_dots(a, b)
    naxb = vec_norms(axb); na = vec_norms(a); nb = vec_norms(b)
    axbu = axb/naxb[..., np.newaxis]
    c = (adb/(na*nb))[..., np.newaxis]; s = (naxb/(na*nb))[..., np.newaxis]
    na = na[..., np.newaxis]; nb = nb[..., np.newaxis]
    au = a / na; bu = b / nb
    dsa = c/na*(-tdo(tdo(e3, bu), axbu) - au*s)
    dca = s/na*(bu - au*c)
    dsb = c/nb*( tdo(tdo(e3, au), axbu) - bu*s)
    dcb = s/nb*(au - bu*c)

    da = dsa-dca
    db = dsb-dcb
    return np.array([-(da+db), da, db])

def dihed_deriv(coords, i, j, k, l):
    """
    Gives the derivative of the dihedral between i, j, k, and l with respect to the Cartesians
    Currently gives what are sometimes called the `psi` angles.
    Will be extended to also support more traditional `phi` angles

    :param coords:
    :type coords: np.ndarray
    :param i:
    :type i: int | Iterable[int]
    :param j:
    :type j: int | Iterable[int]
    :param k:
    :type k: int | Iterable[int]
    :param l:
    :type l: int | Iterable[int]
    :return: derivatives of the dihedral with respect to atoms i, j, k, and l
    :rtype: np.ndarray
    """
    # needs to be vectorized, still

    a = coords[j] - coords[i]
    b = coords[k] - coords[j]
    c = coords[l] - coords[k]

    i3 = np.broadcast_to(np.eye(3), (len(a), 3, 3))
    e3 = np.broadcast_to(levi_cevita3, (len(a), 3, 3, 3))

    tdo = vec_tdot

    # build all the necessary components for the derivatives
    axb = vec_crosses(a, b); bxc = vec_crosses(b, c)
    na = vec_norms(a); nb = vec_norms(b); nc = vec_norms(c)
    naxb = vec_norms(axb); nbxc = vec_norms(bxc)
    naxb = naxb[..., np.newaxis]
    nbxc = nbxc[..., np.newaxis]
    nb = nb[..., np.newaxis]
    n1 = axb / naxb
    n2 = bxc / nbxc
    bu = b / nb
    i3n1 = i3 - vec_outer(n1, n1); i3n2 = i3 - vec_outer(n2, n2)
    dn1a = -np.matmul(vec_tdot(e3, b), i3n1) / naxb[..., np.newaxis]
    dn1b = np.matmul(vec_tdot(e3, a), i3n1) / naxb[..., np.newaxis]
    dn2b = -np.matmul(vec_tdot(e3, c), i3n2) / nbxc[..., np.newaxis]
    dn2c = np.matmul(vec_tdot(e3, b), i3n2) / nbxc[..., np.newaxis]
    dbu = 1/nb[..., np.newaxis]*(i3 - vec_outer(bu, bu))
    n1xb = vec_crosses(bu, n1)
    n1dn2 = vec_dots(n1, n2)
    nxbdn2 = vec_dots(n1xb, n2)

    # compute the actual derivs w/r/t the vectors
    n1dn2 = n1dn2[..., np.newaxis]
    nxbdn2 = nxbdn2[..., np.newaxis]
    dta_1 = tdo(tdo(dn1a, e3), bu)
    dta_2 = tdo(dta_1, n2)*n1dn2
    dta = dta_2 - tdo(dn1a, n2)*nxbdn2
    dtb = (
            ( tdo(tdo(tdo(dn1b, e3), bu)-tdo(tdo(dbu, e3), n1), n2)
              + tdo(dn2b, n1xb) ) * n1dn2
            - (tdo(dn1b, n2) + tdo(dn2b, n1)) * nxbdn2
    )
    dtc = tdo(dn2c, n1xb) * n1dn2 - tdo(dn2c, n1) * nxbdn2

    return np.array([-dta, dta-dtb, dtb-dtc, dtc])

