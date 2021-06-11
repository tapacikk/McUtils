"""
Provides customized set operations based off of the NumPy builtins
to minimize things like excess sorts
"""

import numpy as np

__all__ = [
    'unique',
    'intersection',
    'contained',
    'difference',
    'find',
    'argsort',
    'coerce_dtype'
]

def argsort(ar):
    ar = np.asanyarray(ar)
    if ar.ndim > 1:
        ar, _, _, _ = coerce_dtype(ar)
    return np.argsort(ar, kind='mergesort')

def coerce_dtype(ar, dtype=None):
    """
    Extracted from the way NumPy treats unique
    Coerces ar into a compound dtype so that it can be treated
    like a 1D array for set operations
    """

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    if dtype is None:
        dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]
    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
            if len(consolidated.shape) > 1:
                consolidated = consolidated.squeeze()
                if consolidated.shape == ():
                    consolidated = np.expand_dims(consolidated, 0)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to `coerce_dtype` is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    return consolidated, dtype, orig_shape, orig_dtype

def uncoerce_dtype(consolidated, orig_shape, orig_dtype, axis):
    n = len(consolidated)
    uniq = consolidated.view(orig_dtype)
    uniq = uniq.reshape(n, *orig_shape[1:])
    if axis is not None:
        uniq = np.moveaxis(uniq, 0, axis)
    return uniq

def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=0, sorting=None):
    """
    A variant on np.unique with default support for `axis=0` and sorting
    """

    ar = np.asanyarray(ar)
    if ar.ndim == 1:
        ret = unique1d(ar, return_index=return_index, return_inverse=return_inverse,
                       return_counts=return_counts, sorting=sorting)
        return ret

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.AxisError(axis, ar.ndim)

    # coerce the data into the approrpriate shape
    consolidated, dtype, orig_shape, orig_dtype = coerce_dtype(ar)

    output = unique1d(consolidated,
                                return_index=return_index, return_inverse=return_inverse,
                                return_counts=return_counts, sorting=sorting)
    output = (uncoerce_dtype(output[0], orig_shape, orig_dtype, axis),) + output[1:]
    return output

def unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False, sorting=None):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar)

    if sorting is None:
        sorting = ar.argsort(kind='mergesort') # we want to have stable sorts throughout
    ar = ar[sorting]


    mask = np.empty(ar.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = ar[1:] != ar[:-1]

    ret = (ar[mask], sorting)
    if return_index:
        ret += (sorting[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[sorting] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret

def intersection(ar1, ar2,
                assume_unique=False, return_indices=False,
                sortings=None, union_sorting=None
                ):

    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if ar1.ndim == 1:
        ret = intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices,
                          sortings=sortings, union_sorting=union_sorting)
        return ret

    ar1, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar1)
    ar2, dtype, orig_shape2, orig_dtype2 = coerce_dtype(ar2, dtype=dtype)
    output = intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices,
                          sortings=sortings, union_sorting=union_sorting)
    output = (uncoerce_dtype(output[0], orig_shape1, orig_dtype1, None),) + output[1:]
    return output

def intersect1d(ar1, ar2,
                assume_unique=False, return_indices=False,
                sortings=None, union_sorting=None
                ):
    """
    Find the intersection of two arrays.

    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            if sortings is not None:
                ar1, sorting1, ind1 = unique1d(ar1, return_index=True, sorting=sortings[0])
                ar2, sorting2, ind2 = unique1d(ar2, return_index=True, sorting=sortings[1])
            else:
                ar1, sorting1, ind1 = unique1d(ar1, return_index=True)
                ar2, sorting2, ind2 = unique1d(ar2, return_index=True)
        else:
            if sortings is not None:
                ar1, sorting1 = unique1d(ar1, sorting=sortings[0])
                ar2, sorting2 = unique1d(ar2, sorting=sortings[1])
            else:
                ar1, sorting1 = unique1d(ar1)
                ar2, sorting2 = unique1d(ar2)
        sortings = (sorting1, sorting2)

    aux = np.concatenate((ar1, ar2))
    if union_sorting is None:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux_sort_indices = union_sorting
        aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]
        return int1d, sortings, union_sorting, ar1_indices, ar2_indices
    else:
        return int1d, sortings, union_sorting

def contained(ar1, ar2, assume_unique=False, invert=False,
                sortings=None, union_sorting=None):
    """
    Test whether each element of `ar1` is also present in `ar2`.
    """

    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if ar1.ndim > 1:
        ar1, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar1)
        ar2, dtype, orig_shape2, orig_dtype2 = coerce_dtype(ar2, dtype=dtype)

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask, sortings, union_sorting

    # Otherwise use sorting
    if not assume_unique:
        if sortings is None:
            ar1, sorting1, rev_idx = unique1d(ar1, return_inverse=True)
            ar2, sorting2 = unique1d(ar2)
        else:
            ar1, sorting1, rev_idx = unique1d(ar1, sorting=sortings[0], return_inverse=True)
            ar2, sorting2 = unique1d(ar2, sorting=sortings[1])
        sortings = (sorting1, sorting2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    if union_sorting is None:
        order = ar.argsort(kind='mergesort')
    else:
        order = union_sorting
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)], sortings, order
    else:
        return ret[rev_idx], sortings, order

def difference(ar1, ar2, assume_unique=False, sortings=None, union_sorting=None):
    """
    Calculates set differences over any shape of array
    """

    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if ar1.ndim == 1:
        ret = difference1d(ar1, ar2, assume_unique=assume_unique,
                          sortings=sortings, union_sorting=union_sorting)
        return ret

    ar1, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar1)
    ar2, dtype, orig_shape2, orig_dtype2 = coerce_dtype(ar2, dtype=dtype)
    output = difference1d(ar1, ar2, assume_unique=assume_unique,
                          sortings=sortings, union_sorting=union_sorting)
    output = (uncoerce_dtype(output[0], orig_shape1, orig_dtype1, None),) + output[1:]
    return output

def difference1d(ar1, ar2, assume_unique=False, sortings=None, union_sorting=None):
    """
    Calculates set differences in 1D
    """

    if not assume_unique:
        if sortings is not None:
            ar1, sorting1 = unique(ar1, sorting=sortings[0])
            ar2, sorting2 = unique(ar2, sorting=sortings[0])
        else:
            ar1, sorting1 = unique(ar1)
            ar2, sorting2 = unique(ar2)
        sortings = (sorting1, sorting2)

    in_spec = contained(ar1, ar2, sortings=sortings, union_sorting=union_sorting, assume_unique=True, invert=True)
    return (ar1[in_spec[0]],) + in_spec[1:]

def find1d(ar, to_find, sorting=None, check=True):
    """
    Finds elements in an array and returns sorting
    """

    if sorting is None:
        sorting = np.argsort(ar, kind='mergesort')
    vals = np.searchsorted(ar, to_find, sorter=sorting)
    if isinstance(vals, (np.integer, int)):
        vals = np.array([vals])
    # we have the ordering according to the _sorted_ version of `ar`
    # so now we need to invert that back to the unsorted version
    if len(sorting) > 0:
        big_vals = vals == len(sorting)
        vals[big_vals] = -1
        vals = sorting[vals]
        # now because of how searchsorted works, we need to check if the found values
        # truly agree with what we asked for
        bad_vals = ar[vals] != to_find
        if vals.shape == ():
            if bad_vals:
                vals = -1
        else:
            # print(vals, bad_vals)
            vals[bad_vals] = -1
    else:
        bad_vals = np.full_like(to_find, True)
        vals = np.full_like(vals, -1)
    if check and bad_vals.any():
        raise IndexError("{} not in array".format(to_find[bad_vals]))
    return vals, sorting

def find(ar, to_find, sorting=None, check=True):
    """
    Finds elements in an array and returns sorting
    """

    ar = np.asanyarray(ar)
    to_find = np.asanyarray(to_find)

    if ar.ndim == 1:
        ret = find1d(ar, to_find, sorting=sorting, check=check)
        return ret

    ar, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar)
    to_find, dtype, orig_shape2, orig_dtype2 = coerce_dtype(to_find, dtype=dtype)
    output = find1d(ar, to_find, sorting=sorting, check=check)
    return output

