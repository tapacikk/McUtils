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
levi_cevita3 = [
    [[0, 0, 0], [ 0,0,1],[0,-1,0]],
    [[0, 0,-1], [ 0,0,0],[1, 0,0]],
    [[0, 1, 0], [-1,0,0],[0, 0,0]]
]

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
    :return:
    :rtype:
    """

    # Will still need work to be appropriately vectorized

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
    :type coords:
    :param i:
    :type i:
    :param j:
    :type j:
    :return:
    :rtype:
    """
    v = vec_normalize(coords[j]-coords[i])
    return np.array([-v, v])

def angle_deriv(coords, i, j, k):
    """
    Gives the derivative of the angle between i, j, and k with respect to the Cartesians

    :param coords:
    :type coords:
    :param i:
    :type i:
    :param j:
    :type j:
    :return:
    :rtype:
    """
    a = coords[i] - coords[j]
    b = coords[k] - coords[i]
    e3 = levi_cevita3
    axb = vec_crosses(a, b)
    adb = vec_dots(a, b)
    naxb = vec_norms(axb); na = vec_norms(a); nb = vec_norms(b)
    au = a/na; bu = b/nb
    axbu = axb/naxb
    c = adb/(na*nb); s = naxb/(na*nb)
    dsa = c/na*(-np.dot(np.dot(e3, bu), axbu) - au*s)
    dca = s/na*(bu - au*c)
    dsb = c/nb*( np.dot(np.dot(e3, au), axbu) - bu*s)
    dcb = s/nb*(au - bu*c)

    da = dsa-dca
    db = dsb-dcb
    return np.array([da, -(da+db), db])

def dihed_deriv(coords, i, j, k, l):
    """
    Gives the derivative of the dihedral between i, j, k, and l with respect to the Cartesians

    :param coords:
    :type coords:
    :param i:
    :type i:
    :param j:
    :type j:
    :return:
    :rtype:
    """
    # needs to be vectorized, still

    a = coords[j] - coords[i]
    b = coords[k] - coords[j]
    c = coords[l] - coords[k]
    i3 = np.eye(3); e3 = levi_cevita3

    # build all the necessary components for the derivatives
    axb = vec_crosses(a, b); bxc = vec_crosses(b, c)
    na = vec_norms(a); nb = vec_norms(b); nc = vec_norms(c)
    naxb = vec_norms(axb); nbxc = vec_norms(bxc)
    n1 = axb / naxb; n2 = bxc / nbxc; bu = b / nb
    i3n1 = i3 - vec_outer(n1, n1); i3n2 = i3 - vec_outer(n2, n2)
    dn1a = -np.dot(e3, b, i3n1) / naxb
    dn1b = np.dot(e3, a, i3n1) / naxb
    dn2b = -np.dot(e3, c, i3n2) / nbxc
    dn2c = np.dot(e3, b, i3n2) / nbxc
    dbu = 1 / nb*(i3 - vec_outer(bu, bu))
    n1xb = vec_crosses(bu, n1)
    n1dn2 = np.dot(n1, n2)
    nxbdn2 = np.dot(n1xb, n2)

    # compute the actual derivs w/r/t the vectors
    dta = np.dot(np.dot(dn1a, e3, bu), n2)*n1dn2 - np.dot(dn1a, n2)*nxbdn2
    dtb = (
            ( np.dot(np.dot(dn1b, e3, bu)-np.dot(dbu, e3, n1), n2)
              + np.dot(dn2b, n1xb) ) * n1dn2
            - (np.dot(dn1b, n2) + np.dot(dn2b, n1)) * nxbdn2
    )
    dtc = np.dot(dn2c, n1xb) * n1dn2 - np.dot(dn2c, n1) * nxbdn2
    #Dot[dn2c, n1xb]*Dot[n1, n2] - Dot[dn2c, n1]*Dot[n1xb , n2]

    return np.array([-dta, dta-dtb, dtb-dtc, dtc])

