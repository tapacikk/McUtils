"""
Provides analytic derivatives for some common base terms with the hope that we can reuse them elsewhere
"""
import numpy as np
from .VectorOps import *
from .Options import Options

__all__ = [
    'levi_cevita3',
    'rot_deriv',
    'dist_deriv',
    'angle_deriv',
    'dihed_deriv',
    'vec_norm_derivs',
    'vec_sin_cos_derivs',
    'vec_angle_derivs'
]

# felt too lazy to look up some elegant formula
levi_cevita3 = np.array([
    [
        [ 0,  0,  0],
        [ 0,  0,  1],
        [ 0, -1,  0]
    ],
    [
        [ 0,  0, -1],
        [ 0,  0,  0],
        [ 1,  0,  0]
    ],
    [
        [ 0,  1,  0],
        [-1,  0,  0],
        [ 0,  0,  0]
    ]
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

def vec_norm_derivs(a, order=1, zero_thresh=None):
    """
    Derivative of the norm of `a` with respect to its components

    :param a: vector
    :type a: np.ndarray
    :param order: number of derivatives to return
    :type order: int
    :param zero_thresh:
    :type zero_thresh:
    :return: derivative tensors
    :rtype: list
    """

    if order > 2:
        raise NotImplementedError("derivatives currently only up to order {}".format(2))

    derivs = []

    na = vec_norms(a)
    derivs.append(np.copy(na)) # we return the value itself for Taylor-series reasons

    # print(a.shape)
    a, zeros = vec_handle_zero_norms(a, na, zero_thresh=zero_thresh)
    na = na[..., np.newaxis]
    na[zeros] = Options.zero_placeholder

    if order >= 1:
        d1 = a / na
        # print(a.shape, na.shape)

        derivs.append(d1)

    if order >= 2:
        extra_shape = a.ndim - 1
        if extra_shape > 0:
            i3 = np.broadcast_to(np.eye(3), (1,)*extra_shape + (3, 3))
        else:
            i3 = np.eye(3)
        v = vec_outer(d1, d1)
        # na shold have most of the extra_shape needed
        d2 = (i3 - v) / na[..., np.newaxis]
        derivs.append(d2)

    return derivs

def vec_sin_cos_derivs(a, b, order=1, zero_thresh=None):
    """
    Derivative of `sin(a, b)` and `cos(a, b)` with respect to both vector components

    :param a: vector
    :type a: np.ndarray
    :param a: other vector
    :type a: np.ndarray
    :param order: number of derivatives to return
    :type order: int
    :param zero_thresh: threshold for when a norm should be called 0. for numerical reasons
    :type zero_thresh: None | float
    :return: derivative tensors
    :rtype: list
    """

    if order > 2:
        raise NotImplementedError("derivatives currently only up to order {}".format(2))

    extra_dims = a.ndim - 1

    sin_derivs = []
    cos_derivs = []

    a, n_a = vec_apply_zero_threshold(a, zero_thresh=zero_thresh)
    b, n_b = vec_apply_zero_threshold(b, zero_thresh=zero_thresh)

    n = vec_crosses(a, b)
    n, n_n = vec_apply_zero_threshold(n, zero_thresh=zero_thresh)

    adb = vec_dots(a, b)[..., np.newaxis]
    s = n_n / (n_a * n_b)
    c = adb / (n_a * n_b)

    sin_derivs.append(s)
    cos_derivs.append(c)

    bxn = vec_crosses(b, n)
    bxn, n_bxn = vec_apply_zero_threshold(bxn, zero_thresh=zero_thresh)

    nxa = vec_crosses(n, a)
    nxa, n_nxa = vec_apply_zero_threshold(nxa, zero_thresh=zero_thresh)

    if order <= 1:
        _, na_da = vec_norm_derivs(a, order=1)
        _, nb_db = vec_norm_derivs(b, order=1)
    else:
        _, na_da, na_daa = vec_norm_derivs(a, order=2)
        _, nb_db, nb_dbb = vec_norm_derivs(b, order=2)
        _, nn_dn, nn_dnn = vec_norm_derivs(n, order=2)

    if order >= 1:
        s_da = (bxn / (n_b * n_n) - s * na_da) / n_a
        s_db = (nxa / (n_n * n_a) - s * nb_db) / n_b

        # now we build our derivs, but we also need to transpose so that the OG shape comes first
        d1 = np.array([s_da, s_db])
        d1_reshape = tuple(range(1, extra_dims+1)) + (0, extra_dims+1)
        meh = d1.transpose(d1_reshape)

        sin_derivs.append(meh)

        # print(
        #     nb_db.shape,
        #     na_da.shape,
        #     c.shape,
        #     n_a.shape
        # )

        c_da = (nb_db - c * na_da) / n_a
        c_db = (na_da - c * nb_db) / n_b

        d1 = np.array([c_da, c_db])
        meh = d1.transpose(d1_reshape)

        cos_derivs.append(meh)

    if order >= 2:

        extra_dims = a.ndim - 1
        extra_shape = a.shape[:-1]
        if extra_dims > 0:
            e3 = np.broadcast_to(levi_cevita3,  extra_shape + (3, 3, 3))
            td = np.tensordot
            outer = vec_outer
            vec_td = lambda a, b, **kw: vec_tensordot(a, b, shared=extra_dims, **kw)
        else:
            e3 = levi_cevita3
            td = np.tensordot
            vec_td = lambda a, b, **kw: vec_tensordot(a, b, shared=0, **kw)
            outer = np.outer
            a = a.squeeze()
            nb_db = nb_db.squeeze()
            na_da = na_da.squeeze()
            nn_dn = nn_dn.squeeze()
            na_daa = na_daa.squeeze()
            nb_dbb = nb_dbb.squeeze()
            nn_dnn = nn_dnn.squeeze()

        # compute terms we'll need for various cross-products
        e3b = vec_td(e3, b, axes=[-1, -1])
        e3a = vec_td(e3, a, axes=[-1, -1])
        # e3n = vec_td(e3, n, axes=[-1, -1])

        e3nbdb = vec_td(e3, nb_db, axes=[-1, -1])
        e3nada = vec_td(e3, na_da, axes=[-1, -1])
        e3nndn = vec_td(e3, nn_dn, axes=[-1, -1])

        n_da = -vec_td(e3b,  nn_dnn, axes=[-1, -2])
        bxdna = vec_td(n_da, e3nbdb, axes=[-1, -2])

        # if extra_dims == 0:
        #     print("Good:", bxdna, n_da, e3nbdb)
        # else:
        #     print("Wtf", np.tensordot(n_da[0], e3nbdb[0], axes=[-1, -2]))
        #     print("Bad:", bxdna[0], n_da[0], e3nbdb[0], e3nbdb[1])

        s_daa = - (
            outer(na_da, s_da) + outer(s_da, na_da) + s[..., np.newaxis] * na_daa - bxdna
        ) / n_a[..., np.newaxis]

        ndaXnada = -vec_td(n_da, e3nada, axes=[-1, -2])
        nndnXnadaa = vec_td(na_daa, e3nndn, axes=[-1, -2])

        s_dab = (
                     ndaXnada + nndnXnadaa - outer(s_da, nb_db)
        ) / n_b[..., np.newaxis]

        n_db = vec_td(e3a, nn_dnn, axes=[-1, -2])

        nbdbXnda = vec_td(n_db, e3nbdb, axes=[-1, -2])
        nbdbbXnndn = -vec_td(nb_dbb, e3nndn, axes=[-1, -2])

        s_dba = (
                nbdbXnda + nbdbbXnndn - outer(s_db, na_da)
        ) / n_a[..., np.newaxis]

        dnbxa = - vec_td(n_db, e3nada, axes=[-1, -2])

        s_dbb = - (
            outer(nb_db, s_db) + outer(s_db, nb_db) + s[..., np.newaxis] * nb_dbb - dnbxa
        ) / n_b[..., np.newaxis]

        s2 = np.array([
            [s_daa, s_dab],
            [s_dba, s_dbb]
        ])

        d2_reshape = tuple(range(2, extra_dims+2)) + (0, 1, extra_dims+2, extra_dims+3)
        # raise Exception(s2.shape, d2_reshape)
        s2 = s2.transpose(d2_reshape)

        sin_derivs.append(s2)

        c_daa = - (
            outer(na_da, c_da) + outer(c_da, na_da) + c[..., np.newaxis] * na_daa
        ) / n_a[..., np.newaxis]

        c_dab = ( na_daa - outer(c_da, nb_db) ) / n_b[..., np.newaxis]

        c_dba = ( nb_dbb - outer(c_db, na_da) ) / n_a[..., np.newaxis]

        c_dbb = - (
            outer(nb_db, c_db) + outer(c_db, nb_db) + c[..., np.newaxis] * nb_dbb
        ) / n_b[..., np.newaxis]

        c2 = np.array([
            [c_daa, c_dab],
            [c_dba, c_dbb]
        ])

        c2 = c2.transpose(d2_reshape)

        # raise Exception(c2.shape)

        cos_derivs.append(c2)

    return sin_derivs, cos_derivs

def vec_angle_derivs(a, b, order=1, zero_thresh=None):
    """
    Returns the derivatives of the angle between `a` and `b` with respect to their components

    :param a: vector
    :type a: np.ndarray
    :param b: vector
    :type b: np.ndarray
    :param order: order of derivatives to go up to
    :type order: int
    :param zero_thresh: threshold for what is zero in a vector norm
    :type zero_thresh: float | None
    :return: derivative tensors
    :rtype: list
    """

    if order > 2:
        raise NotImplementedError("derivatives currently only up to order {}".format(2))

    derivs = []

    sin_derivs, cos_derivs = vec_sin_cos_derivs(a, b, order=order, zero_thresh=zero_thresh)

    s = sin_derivs[0]
    c = cos_derivs[0]

    q = np.arctan2(s, c)

    derivs.append(q)

    if order >= 1:
        s_da, s_db = sin_derivs[1]
        c_da, c_db = cos_derivs[1]

        q_da = c * s_da - s * c_da
        q_db = c * s_db - s * c_db

        derivs.append(np.array([q_da, q_db]))

    if order >= 2:
        s_daa, s_dab = sin_derivs[2][0]
        s_dba, s_dbb = sin_derivs[2][1]
        c_daa, c_dab = cos_derivs[2][0]
        c_dba, c_dbb = cos_derivs[2][1]

        q_daa = vec_outer(c_da, s_da) + c * s_daa - vec_outer(s_da, c_da) - s * c_daa
        q_dba = vec_outer(c_da, s_db) + c * s_dba - vec_outer(s_da, c_db) - s * c_dba
        q_dab = vec_outer(c_db, s_da) + c * s_dab - vec_outer(s_db, c_da) - s * c_dab
        q_dbb = vec_outer(c_db, s_db) + c * s_dbb - vec_outer(s_db, c_db) - s * c_dbb

        derivs.append(np.array([
            [q_daa, q_dab],
            [q_dba, q_dbb]
        ]))

    return derivs

def dist_deriv(coords, i, j, order=1, zero_thresh=None):
    """
    Gives the derivative of the distance between i and j with respect to coords i and coords j

    :param coords:
    :type coords: np.ndarray
    :param i: index of one of the atoms
    :type i: int | Iterable[int]
    :param j: index of the other atom
    :type j: int | Iterable[int]
    :return: derivatives of the distance with respect to atoms i, j, and k
    :rtype: list
    """

    if order > 2:
        raise NotImplementedError("derivatives currently only up to order {}".format(2))

    a = coords[j] - coords[i]
    d = vec_norm_derivs(a, order=order, zero_thresh=zero_thresh)

    derivs = []

    derivs.append(d[0])

    if order >= 1:
        da = d[1]
        derivs.append(np.array([-da, da]))

    if order >= 2:
        daa = d[2]
        # ii ij
        # ji jj
        derivs.append(np.array([
            [ daa, -daa],
            [-daa,  daa]
        ]))

    return derivs

def angle_deriv(coords, i, j, k, order=1, zero_thresh=None):
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

    if order > 2:
        raise NotImplementedError("derivatives currently only up to order {}".format(2))

    a = coords[j] - coords[i]
    b = coords[k] - coords[i]
    d = vec_angle_derivs(a, b, order=order, zero_thresh=zero_thresh)

    derivs = []

    derivs.append(d[0])

    if order >= 1:
        da, db = d[1]
        derivs.append(np.array([-(da + db), da, db]))

    if order >= 2:
        daa, dab = d[2][0]
        dba, dbb = d[2][1]

        # raise Exception(d[2])
        # ii ij ik
        # ji jj jk
        # ki kj kk
        derivs.append(np.array([
            [daa + dba + dab + dbb, -(daa + dab), -(dba + dbb)],
            [         -(daa + dba),          daa,   dba       ],
            [         -(dab + dbb),          dab,   dbb       ]
        ]))

    return derivs

def dihed_deriv(coords, i, j, k, l, order=1, zero_thresh=None):
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

    if order > 2:
        raise NotImplementedError("derivatives currently only up to order {}".format(2))

    a = coords[j] - coords[i]
    b = coords[k] - coords[j]
    c = coords[l] - coords[k]

    n1 = vec_crosses(a, b)
    n2 = vec_crosses(b, c)

    sign = np.sign(vec_dots(b, vec_crosses(n1, n2)))

    d = vec_angle_derivs(n1, n2, order=order, zero_thresh=zero_thresh)

    derivs = []

    derivs.append(d[0])

    i3 = np.broadcast_to(np.eye(3), (len(a), 3, 3))
    e3 = np.broadcast_to(levi_cevita3, (len(a), 3, 3, 3))

    if order >= 1:
        dn1, dn2 = d[1]

        di = vec_crosses(b, dn1)
        dj = vec_crosses(c, dn2) - vec_crosses(a+b, dn1)
        dk = vec_crosses(a, dn1) - vec_crosses(b+c, dn2)
        dl = vec_crosses(b, dn2)

        derivs.append(sign[np.newaxis, ..., np.newaxis]*np.array([di, dj, dk, dl]))

    if order >= 2:
        d11, d12 = d[2][0]
        d21, d22 = d[2][1]

        # explicit write out of chain-rule transformations to isolate different Kron-delta terms
        dot = vec_tensordot

        Ca = dot(e3, a)
        Cb = dot(e3, b)
        Cc = dot(e3, c)
        Cab = Ca+Cb
        Cbc = Cb+Cc

        CaCa, CaCb, CaCc, CaCab, CaCbc = [dot(Ca, x) for x in (Ca, Cb, Cc, Cab, Cbc)]
        CbCa, CbCb, CbCc, CbCab, CbCbc = [dot(Cb, x) for x in (Ca, Cb, Cc, Cab, Cbc)]
        CcCa, CcCb, CcCc, CcCab, CcCbc = [dot(Cc, x) for x in (Ca, Cb, Cc, Cab, Cbc)]
        CabCa, CabCb, CabCc, CabCab, CabCbc = [dot(Cab, x) for x in (Ca, Cb, Cc, Cab, Cbc)]
        CbcCa, CbcCb, CbcCc, CbcCab, CbcCbc = [dot(Cab, x) for x in (Ca, Cb, Cc, Cab, Cbc)]

        dii = dot(CbCb, d11)
        dij = dot(CbCc, d21) - dot(CbCab, d11)
        dik = dot(CbCa, d11) - dot(CbCbc, d21)
        dil = dot(CbCb, d21)

        dji = dot(CcCb, d12) - dot(CabCb, d11)
        djj = dot(CabCab, d11) - dot(CabCc, d21) - dot(CcCab, d12) + dot(CcCc, d22)
        djk = dot(CabCbc, d21) - dot(CcCbc, d22) - dot(CabCa, d11) + dot(CcCa, d12)
        djl = dot(CcCb, d22) - dot(CabCb, d21)

        dki = dot(CaCb, d11) - dot(CbcCb, d12)
        dkj = dot(CbcCab, d12) - dot(CbcCc, d22) - dot(CaCab, d11) + dot(CaCc, d21)
        dkk = dot(CaCa, d11) - dot(CaCbc, d21) - dot(CbcCa, d12) + dot(CbcCbc, d22)
        dkl = dot(CaCb, d21) - dot(CbcCb, d22)

        dli = dot(CbCb, d12)
        dlj = dot(CbCc, d22) - dot(CbCab, d12)
        dlk = dot(CbCa, d12) - dot(CbCbc, d22)
        dll = dot(CbCb, d22)

        derivs.append(sign[np.newaxis, np.newaxis, ..., np.newaxis]**np.array([
            [dii, dij, dik, dil],
            [dji, djj, djk, djl],
            [dki, dkj, dkk, dkl],
            [dli, dlj, dlk, dll]
        ]))

    return derivs

# debug implementations
# def dist_deriv(coords, i, j, order=1):
#     """
#     Gives the derivative of the distance between i and j with respect to coords i and coords j
#     :param coords:
#     :type coords: np.ndarray
#     :param i: index of one of the atoms
#     :type i: int | Iterable[int]
#     :param j: index of the other atom
#     :type j: int | Iterable[int]
#     :return: derivatives of the distance with respect to atoms i, j, and k
#     :rtype: np.ndarray
#     """
#     v = vec_normalize(coords[j]-coords[i])
#
#     return None, np.array([-v, v])

# def angle_deriv(coords, i, j, k, order=1):
#     """
#     Gives the derivative of the angle between i, j, and k with respect to the Cartesians
#     :param coords:
#     :type coords: np.ndarray
#     :param i: index of the central atom
#     :type i: int | Iterable[int]
#     :param j: index of one of the outside atoms
#     :type j: int | Iterable[int]
#     :param k: index of the other outside atom
#     :type k: int | Iterable[int]
#     :return: derivatives of the angle with respect to atoms i, j, and k
#     :rtype: np.ndarray
#     """
#
#     dot = vec_dots
#     tdo = vec_tdot
#     a = coords[j] - coords[i]
#     b = coords[k] - coords[i]
#     e3 = np.broadcast_to(levi_cevita3, (len(a), 3, 3, 3))
#     axb = vec_crosses(a, b)
#     adb = vec_dots(a, b)
#     naxb = vec_norms(axb); na = vec_norms(a); nb = vec_norms(b)
#     axbu = axb/naxb[..., np.newaxis]
#     c = (adb/(na*nb))[..., np.newaxis]; s = (naxb/(na*nb))[..., np.newaxis]
#     na = na[..., np.newaxis]; nb = nb[..., np.newaxis]
#     au = a / na; bu = b / nb
#     dsa = c/na*(-tdo(tdo(e3, bu), axbu) - au*s)
#     dca = s/na*(bu - au*c)
#     dsb = c/nb*( tdo(tdo(e3, au), axbu) - bu*s)
#     dcb = s/nb*(au - bu*c)
#
#     da = dsa-dca
#     db = dsb-dcb
#     return None, np.array([-(da+db), da, db])
#
# def dihed_deriv(coords, i, j, k, l, order=1):
#     """
#     Gives the derivative of the dihedral between i, j, k, and l with respect to the Cartesians
#     Currently gives what are sometimes called the `psi` angles.
#     Will be extended to also support more traditional `phi` angles
#     :param coords:
#     :type coords: np.ndarray
#     :param i:
#     :type i: int | Iterable[int]
#     :param j:
#     :type j: int | Iterable[int]
#     :param k:
#     :type k: int | Iterable[int]
#     :param l:
#     :type l: int | Iterable[int]
#     :return: derivatives of the dihedral with respect to atoms i, j, k, and l
#     :rtype: np.ndarray
#     """
#     # needs to be vectorized, still
#
#     a = coords[j] - coords[i]
#     b = coords[k] - coords[j]
#     c = coords[l] - coords[k]
#
#     i3 = np.broadcast_to(np.eye(3), (len(a), 3, 3))
#     e3 = np.broadcast_to(levi_cevita3, (len(a), 3, 3, 3))
#
#     tdo = vec_tdot
#
#     # build all the necessary components for the derivatives
#     axb = vec_crosses(a, b); bxc = vec_crosses(b, c)
#     na = vec_norms(a); nb = vec_norms(b); nc = vec_norms(c)
#     naxb = vec_norms(axb); nbxc = vec_norms(bxc)
#     naxb = naxb[..., np.newaxis]
#     nbxc = nbxc[..., np.newaxis]
#     nb = nb[..., np.newaxis]
#     n1 = axb / naxb
#     n2 = bxc / nbxc
#     bu = b / nb
#     i3n1 = i3 - vec_outer(n1, n1); i3n2 = i3 - vec_outer(n2, n2)
#     dn1a = -np.matmul(vec_tdot(e3, b), i3n1) / naxb[..., np.newaxis]
#     dn1b = np.matmul(vec_tdot(e3, a), i3n1) / naxb[..., np.newaxis]
#     dn2b = -np.matmul(vec_tdot(e3, c), i3n2) / nbxc[..., np.newaxis]
#     dn2c = np.matmul(vec_tdot(e3, b), i3n2) / nbxc[..., np.newaxis]
#     dbu = 1/nb[..., np.newaxis]*(i3 - vec_outer(bu, bu))
#     n1xb = vec_crosses(bu, n1)
#     n1dn2 = vec_dots(n1, n2)
#     nxbdn2 = vec_dots(n1xb, n2)
#
#     # compute the actual derivs w/r/t the vectors
#     n1dn2 = n1dn2[..., np.newaxis]
#     nxbdn2 = nxbdn2[..., np.newaxis]
#     dta_1 = tdo(tdo(dn1a, e3), bu)
#     dta_2 = tdo(dta_1, n2)*n1dn2
#     dta = dta_2 - tdo(dn1a, n2)*nxbdn2
#     dtb = (
#             ( tdo(tdo(tdo(dn1b, e3), bu)-tdo(tdo(dbu, e3), n1), n2)
#               + tdo(dn2b, n1xb) ) * n1dn2
#             - (tdo(dn1b, n2) + tdo(dn2b, n1)) * nxbdn2
#     )
#     dtc = tdo(dn2c, n1xb) * n1dn2 - tdo(dn2c, n1) * nxbdn2
#
#     return None, np.array([-dta, dta-dtb, dtb-dtc, dtc])
