
import itertools, numpy as np
from .VectorOps import vec_tensordot, vec_outer

__all__ = [
    "nca_op_deriv",
    "tensordot_deriv",
    "tensorprod_deriv"
]

def get_nca_shifts(order, k):
    permute_pos = np.arange(order, order+k)
    ncombs = np.math.comb(order, k)
    shifts = np.broadcast_to(np.arange(order)[np.newaxis], (ncombs, order)).copy()
    for i,pos in enumerate(itertools.combinations(range(order), r=k)):
        shifts[i, pos] = permute_pos
    return shifts
def apply_nca_op(op, order, k, A_expansion, B_expansion, deriv_axis, a, b, contract, shared):
    s = order - k
    if s >= len(A_expansion) or k >= len(B_expansion):
        return 0

    A = A_expansion[s]
    B = B_expansion[k]
    if shared is None:
        shared = 0
    if contract: # axes disappear, so we just account for the shifts
        axes = [[x+s for x in a], [x+k for x in b]]
    else: # axes appeared, so we need to include those in the product
          # we _forced_ axes to be at the end of the arrays since it was too hard
          # to keep track of them otherwise...
          # actually I guess I could have put the derivative axes at the end...
          # and then the axes would never change...but that has other complications
        axes = [
            [shared + i for i in range(s)] + [x+s for x in a],
            [shared + i for i in range(k)] + [x+k for x in b]
        ]
    if shared == 0:
        base = op(A, B, axes=axes)
    else:
        # print(A.shape, B.shape, axes, shared)
        base = op(A, B, axes=axes, shared=shared)

    # now we do the necessary permutations
    full = None
    for shift in get_nca_shifts(order, k):
        sub = base
        for i,x in enumerate(shift):
            if i != x:
                sub = np.moveaxis(sub, deriv_axis + x-order, shared + i) # nA includes the shared dimensions
        if full is None:
            full = sub
        else:
            full = full + sub

    return full
def nca_op_order_deriv(op, order, A_expansion, B_expansion, deriv_axis, a, b, contract, shared):
    full = None
    for k in range(order+1):
        term = apply_nca_op(op, order, k, A_expansion, B_expansion, deriv_axis, a, b, contract, shared)
        if full is None:
            full = term
        else:
            full = full + term
    return full
def nca_op_deriv(op,
                 A_expansion,
                 B_expansion,
                 order,
                 axes,
                 contract,
                 shared=None
                 ):
    A_expansion = [np.asanyarray(A) for A in A_expansion]
    B_expansion = [np.asanyarray(B) for B in B_expansion]

    a_ax, b_ax = axes
    if isinstance(a_ax, int): a_ax = [a_ax]
    if isinstance(b_ax, int): b_ax = [b_ax]
    a_ax = [ a if a >= 0 else A_expansion[0].ndim + a for a in a_ax ]
    b_ax = [ b if b >= 0 else B_expansion[0].ndim + b for b in b_ax ]

    if contract: # the derivative axis will always be at nA + 1 - num_contracted - the shared axes
        deriv_axis = A_expansion[0].ndim + 1 - len(axes)
    else:
        # we require that the outer product be ordered
        # so we now which axes to move around
        a_ax = np.sort(a_ax)
        a_dim = A_expansion[0].ndim
        if np.any(a_ax != np.arange(a_dim - len(a_ax), a_dim)):
            raise ValueError("axes {} must be the final axes of A".format(a_ax))
        b_ax = np.sort(b_ax)
        b_dim = B_expansion[0].ndim
        if np.any(b_ax != np.arange(b_dim - len(b_ax), b_dim)):
            raise ValueError("axes {} must be the final axes of B".format(b_ax))
        deriv_axis = a_dim

    if isinstance(order, int):
        order = list(range(1, order+1))

    # if shared is not None:
    #     deriv_axis = deriv_axis - shared

    derivs = [
        nca_op_order_deriv(op, o, A_expansion, B_expansion, deriv_axis, a_ax, b_ax, contract, shared)
        for o in order
    ]

    return derivs

def tensordot_deriv(A_expansion, B_expansion,
                    order,
                    axes=None,
                    shared=None
                    ):

    if axes is None: axes = [-1, 0]
    if shared is not None:
        op = vec_tensordot
    else:
        op = np.tensordot

    return nca_op_deriv(op,
                        A_expansion, B_expansion, order,
                        axes=axes,
                        contract=True,
                        shared=shared
                        )

def tensorprod_deriv(A_expansion, B_expansion,
                    order,
                    axes=None
                    ):

    if axes is None:
        axes = [-1, -1]
    a_ax, b_ax = axes
    if isinstance(a_ax, int): a_ax = [a_ax]
    if isinstance(b_ax, int): b_ax = [b_ax]

    shared = A_expansion[0].ndim - len(a_ax)

    return nca_op_deriv(
        lambda left, right, axes=None, shared=None: vec_outer(left, right, axes=axes),
        A_expansion, B_expansion, order,
        axes=[a_ax, b_ax],
        contract=False,
        shared=shared
    )
