import numpy as np, scipy.sparse as sp, itertools as ip, functools as fp

__all__ = [
    "SparseArray",
    "sparse_tensordot"
]

#   I should provide a general "ArrayWrapper" class that allows all these arrays to interact nicely
#   with eachother with a very general `dot` syntax and shared `reshape` interface and all of that
#   that way I would never need to check if things are numpy.ndarray or sp.spmatrix or those kinds of things
#
#   And actually by providing a constructor like np.array that dispatches to the appropriate class
#   for a given data type we'd be able to avoid the need to explicitly declare our data-type

class SparseArray:
    """
    Array class that generalize the regular `scipy.sparse`
    Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.
    """

    def __init__(self, a, shape = None, layout = sp.csc_matrix, initialize = True):
        self._shape = tuple(shape) if shape is not None else shape
        self._a = a
        self._fmt = layout
        self._validated = False
        if initialize:
            self._init_matrix()
            self._validate()
        self._block_inds = None # cached to speed things up?
        self._block_vals = None # cached to speed things up?
    def _init_matrix(self):
        a = self._a
        if isinstance(a, sp.spmatrix):
            if self._shape is None:
                self._shape = a.shape
        elif isinstance(a, np.ndarray):
            self._shape = a.shape
            if len(a.shape) > 2:
                a = a.reshape((a.shape[0], np.prod(a.shape[1:])))
            elif len(a.shape) == 1:
                a = a.reshape(a.shape + (1,))
            self._a = self.fmt(a, shape=a.shape)
        else:
            non_sparse, sparse = self._get_shape()
            if non_sparse is None:
                self._a = np.array(self._a)
                self._init_matrix()
            else:
                self._shape = non_sparse + sparse
                data, inds, total_shape = self._get_data(non_sparse, sparse)
                self._a = data
                flat = np.ravel_multi_index(inds, data.shape)
                self._block_inds = flat, inds
    def _validate(self):
        shp1 = self._a.shape
        shp2 = self._shape
        if np.prod(shp1) != np.prod(shp2):
            raise ValueError("sparse data with shape {} cannot correspond to shape {}".format(
                shp1,
                shp2
            ))
        self._validated = True
    def _get_shape(self):
        non_sp = []
        elm = self._a
        while not isinstance(elm, sp.spmatrix):
            try:
                elm_2 = elm[0]
            except (TypeError, IndexError):
                return None, None
            else:
                non_sp.append(len(elm))
                elm = elm_2
        else: # only catch if no break
            shape = elm.shape

        non_sparse_shape = tuple(non_sp)
        return non_sparse_shape, tuple(shape)
    def _get_data(self, non_sparse, sparse):
        """
        We'll take our blocks and arrange them as a vector of sparse arrays
        """
        inds = ip.product(*(list(range(x)) for x in non_sparse))
        a = self._a
        bits = np.cumprod(non_sparse) # so that we can build our vector of blocks
        if len(sparse) == 1:
            sparse = sparse+(1,)
        def pull(a, idx, bits=bits, shp = sparse):
            block = fp.reduce(lambda b,i:b[i], idx, a)
            if not isinstance(block, (sp.spmatrix, np.ndarray)):
                block = np.array(block)
            block_data, row_ind, col_ind = sp.find(block)
            offset = np.dot(idx, bits)*shp[1]
            return block_data, row_ind, col_ind + offset

        blocks = [pull(a, idx) for idx in inds]
        block_data = np.concatenate([b[2] for b in blocks])
        row_inds = np.concatenate([b[0] for b in blocks])
        col_inds = np.concatenate([b[1] for b in blocks])
        total_shape = (sparse[0], bits[-1]*sparse[1])
        inds = (row_inds, col_inds)
        data = self._build_data(block_data, inds, total_shape)
        return data, (block_data, inds, total_shape)
    def _build_data(self, block_data, inds, total_shape):
        try:
            data = self.fmt((block_data, inds), shape=total_shape)
        except (ValueError, TypeError):
            data = self.fmt(sp.csc_matrix((block_data, inds), shape=total_shape))
        return data

    @classmethod
    def from_diag(cls, diags, **kw):
        data = sp.block_diag(diags, format='csr')
        block_size = diags[0].shape
        if 'shape' not in kw:
            kw['shape'] = (len(diags), block_size[0], len(diags), block_size[1])
        return cls(data, **kw).transpose((0, 2, 1, 3))

    def toarray(self):
        return np.reshape(self.data.toarray(), self.shape)
    def todense(self):
        return np.reshape(np.asarray(self.data.todense()), self.shape)
    @property
    def data(self):
        if not isinstance(self._a, sp.spmatrix):
            self._init_matrix()
        if not self._validated:
            self._validate()
        return self._a
    @property
    def fmt(self):
        return self._fmt
    @property
    def shape(self):
        if self._shape is None:
            self._init_matrix()
        if not self._validated:
            self._validate()
        return self._shape
    @property
    def ndim(self):
        return len(self.shape)

    # this saves time when we have to do a bunch of reshaping into similarly sized arrays,
    # but won't help as much when the shape changes
    _unravel_cache = {}  # hopefully faster than bunches of unravel_index calls...
    @classmethod
    def _unravel_indices(cls, n, dims):
        # we're hoping that we call with `n` often enough that we get a performance benefit
        if dims not in cls._unravel_cache:
            cls._unravel_cache[dims] = {}
        cache = cls._unravel_cache[dims]
        if isinstance(n, np.ndarray):
            n_hash = hash(n.data.tobytes())
        else:
            n_hash = hash(n)
        if n_hash in cache:
            res = cache[n_hash]
        else:
            res = np.unravel_index(n, dims)
            cache[n_hash] = res
        return res
    @property
    def block_vals(self):
        if True:#self._block_vals is None:
            d = self.data
            row_inds, col_inds, data = sp.find(d)
            flat = np.ravel_multi_index((row_inds, col_inds), d.shape)
            unflat = self._unravel_indices(flat, self.shape)
            self._block_inds = (flat, unflat)
            self._block_vals = data
        return self._block_vals
    @property
    def block_inds(self):
        if True:#self._block_inds is None:
            vals = self.block_vals
        return self._block_inds
    @block_inds.setter
    def block_inds(self, bi):
        if isinstance(bi, tuple) and len(bi) == 2:
            flat, unflat = bi
            if isinstance(unflat[0], int):
                row_inds, col_inds = bi
                flat = np.ravel_multi_index((row_inds, col_inds), self.data.shape)
                unflat = self._unravel_indices(flat, self.shape)
        elif isinstance(bi[0], int):
            flat = bi
            unflat = self._unravel_indices(flat, self.shape)
        else:
            unflat = bi
            flat = np.ravel_multi_index(bi, self.shape)
        self._block_inds = (flat, unflat)
    @property
    def block_data(self):
        return self.block_vals, self.block_inds[1]

    def transpose(self, transp):
        """
        Transposes the array and returns a new one.
        Not necessarily a cheap operation.

        :param transp: the transposition to do
        :type transp: Iterable[int]
        :return:
        :rtype:
        """
        shp = self.shape
        data, inds = self.block_data
        new_inds = [inds[i] for i in transp]
        new_shape = [shp[i] for i in transp]
        if len(new_shape) > 2:
            total_shape = (np.prod(new_shape[:-2])*new_shape[-2], new_shape[-1])
            flat = np.ravel_multi_index(new_inds, new_shape)
            unflat = self._unravel_indices(flat, total_shape)
        else:
            flat = None
            unflat = new_inds
            total_shape = new_shape
        data = self._build_data(data, unflat, total_shape)
        new = type(self)(data, shape = new_shape, layout = self.fmt)
        if self._block_vals is not None:
            # there's got to be a way to map the old indices onto the new ones & hence sort stuff...
            # and the trick is that _unflat_ ends up being sorted and so we need to first sort
            #
            arr = np.lexsort(unflat)
            new_v = self._block_vals[arr]
            # find_old = sp.find(self.data)
            # old_flat = np.ravel_multi_index(find_old[:2], self.data.shape)
            # find_new = sp.find(data)
            # new_flat = np.ravel_multi_index(find_new[:2], new.data.shape)
            # find_v = find_new[2]
            # raise Exception(
            #     np.max(np.abs(find_v - new_v))
            #     # find_new[1][100:110],
            #     # unflat[1][100:110],
            #     # np.max(np.abs(np.sort(unflat[1]) - np.sort(find_new[1]))),
            #     # np.max(np.abs(np.sort(unflat[0]) - np.sort(find_new[0]))),
            #     # np.max(np.abs(new_flat - flat)),
            #     # old_flat[100:110], flat[100:110],
            #     # find_v[100:110], self._block_vals[100:110]
            # )
            new._block_vals = new_v
        if flat is None:
            new.block_inds = unflat
        else:
            new.block_inds = (flat, unflat)
        return new

    def reshape(self, shp):
        # this is an in-place operation which feels kinda dangerous?
        if np.prod(shp) != np.prod(self.shape):
            raise ValueError("Can't reshape {} into {}".format(self.shape, shp))
        self._shape = tuple(shp)
        bi = self._block_inds
        if bi is not None:
            flat, unflat = bi
            self._block_inds = flat
        return self
    def squeeze(self):
        self.reshape([x for x in self.shape if x != 1])
        return self

    def __matmul__(self, other):
        return self.dot(other)
    def dot(self, b, reverse=False):
        if reverse:
            return sparse_tensordot(b, self, axes=(-1, 1))
        else:
            return sparse_tensordot(self, b, axes=(-1, 1))
    def tensordot(self, b, axes=2, reverse=False):
        if reverse:
            return sparse_tensordot(b, self, axes=axes)
        else:
            return sparse_tensordot(self, b, axes=axes)

    def __add__(self, other):
        return self.plus(other)
    def plus(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) and other == 0.:
            return self
        d = self.data
        if isinstance(other, SparseArray):
            # should do error checking to make sure the shapes work out...
            other = other.data.reshape(d.shape)
        new = d+other
        if isinstance(new, sp.spmatrix):
            return type(self)(new, shape=self.shape, layout=self.fmt)
        else:
            return np.array(new).reshape(self.shape)

    def floopy_flop(self):
        return type(self)(1/self.data, shape = self.shape, layout=self.fmt)
    def __truediv__(self, other):
        return self.multiply(1/other)
    def __rtruediv__(self, other):
        if other == 1:
            return self.floopy_flop()
        return self.multiply(1/other)
    def __rmul__(self, other):
        return self.multiply(other)
    def __mul__(self, other):
        return self.multiply(other)
    def multiply(self, other):
        d = self.data
        if isinstance(other, (int, float, np.integer, np.floating)) and other == 0.:
            new = self.fmt(d.shape)
            return type(self)(new, shape=self.shape, layout=self.fmt)

        if isinstance(other, SparseArray):
            other = other.data
        if isinstance(other, sp.spmatrix):
            other = other.reshape(d.shape)
        elif isinstance(other, np.ndarray):
            other = np.broadcast_to(other, self.shape).reshape(d.shape)
        new = d.multiply(other)
        if isinstance(new, sp.spmatrix):
            return type(self)(new, shape=self.shape, layout=self.fmt)
        else:
            return np.array(new).reshape(self.shape)

    def _get_element(self, idx):
        """
        Convert idx into a 1D index or slice or whatever and then convert it back to the appropriate 2D shape
        :param i:
        :type i:
        :return:
        :rtype:
        """

        # print(">>>>", idx, self.shape)
        if isinstance(idx, (int, np.integer)):
            idx = (idx,)
        pull_elements = len(idx) == len(self.shape) and all(isinstance(x, (int, np.integer)) for x in idx)
        if not pull_elements:
            pull_elements = all(not isinstance(x, (int, np.integer, slice)) for x in idx)
            if pull_elements:
                e1 = len(idx[0])
                pull_elements = all(len(x) == e1 for x in idx)
        if pull_elements:
            flat = np.ravel_multi_index(idx, self.shape)
            unflat = self._unravel_indices(flat, self.data.shape)
            res = self.data[unflat]
            if not isinstance(flat, int):
                res = np.array(res)
            return res
        else:
            # need to compute the shape of the resultant block _and_ keep it as
            blocks = [np.array([i]) if isinstance(i, (int, np.integer)) else np.arange(s)[i] for i, s in zip(idx, self.shape)]
            # print(blocks)
            new_shape = [len(x) for x in blocks if len(x) > 1] + list(self.shape[len(blocks):])
            # print(new_shape)
            data, inds = self.block_data
            inds = list(inds)
            for i, b, s in zip(range(len(blocks)), blocks, self.shape):
                k = 0
                def g(b, j):
                    w = np.argwhere(b == j)
                    if len(w) > 0:
                        w = w.flatten()[0]
                    else:
                        w = -1
                    return w
                mapping = np.array([g(b, j) for j in range(s)])
                ixs = inds[i]
                filter = np.sum(ixs == j for j in b).astype(bool)
                inds = [ix[filter] for ix in inds]
                inds[i] = mapping[inds[i]]
                data = data[filter]

            inds = [ix for ix,j in zip(inds, new_shape) if j > 1 ]
            new_shape = [j for j in new_shape if j > 1]

            if len(new_shape) > 2:
                total_shape = (np.prod(new_shape[:-2]) * new_shape[-2], new_shape[-1])
                flat = np.ravel_multi_index(inds, new_shape)
                unflat = self._unravel_indices(flat, total_shape)
            elif len(new_shape) == 1:
                flat = None
                unflat = inds+[np.zeros((len(inds[0]),))]
                total_shape = new_shape+[1]
            else:
                flat = None
                unflat = inds
                total_shape = new_shape

            try:
                data = self._build_data(data, unflat, total_shape)
            except Exception as e:
                # print(data.shape, unflat)
                data = e
            if isinstance(data, Exception):
                raise IndexError("{}: couldn't take element {} of array {} (Got Error: '{}')".format(
                    type(self).__name__,
                    idx,
                    self,
                    data
                ))
            new = type(self)(data, shape=new_shape, layout=self.fmt)
            # if flat is None:
            #     new.block_inds = unflat
            # else:
            #     new.block_inds = flat, unflat
            return new
    def __getitem__(self, item):
        return self._get_element(item)

    def __repr__(self):
        return "{}(<{}>)".format(type(self).__name__, ", ".join([str(x) for x in self.shape]))

def _dot(a, b):
    if isinstance(a, SparseArray):
        a = a.data
    if isinstance(b, SparseArray):
        b = b.data
    if isinstance(a, sp.spmatrix):
        return a.dot(b)
    else:
        return np.dot(a, b)
def asCOO(a):
    if not isinstance(a, sp.coo_matrix):
        a = sp.coo_matrix(a)
    return a
def sparse_tensordot(a, b, axes=2):
    """Defines a version of tensordot that uses sparse arrays, adapted from the sparse package on PyPI

    :param a: the array to contract from
    :type a: SparseArray | sp.spmatrix | np.ndarray
    :param b: the array to contract with
    :type b: SparseArray | sp.spmatrix | np.ndarray
    :param axes: the axes to contract along
    :type axes: int | Iterable[int]
    :return:
    :rtype:
    """

    try:
        iter(axes)
    except TypeError:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    oldb = [bs[axis] for axis in notin]

    if any(dim == 0 for dim in ip.chain(newshape_a, newshape_b)):
        res = asCOO(np.empty(olda + oldb))
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            res = res.todense()
        return res

    at = a.transpose(newaxes_a)
    if isinstance(at, SparseArray):
        at = at.data
    at = at.reshape(newshape_a)
    bt = b.transpose(newaxes_b)
    if isinstance(bt, SparseArray):
        bt = bt.data
    bt = bt.reshape(newshape_b)
    res = _dot(at, bt)
    if isinstance(res, sp.spmatrix):
        if isinstance(a, SparseArray):
            res = SparseArray(res, shape=olda + oldb, layout=a.fmt)
        elif isinstance(b, sp.spmatrix):
            res = SparseArray(res, shape=olda + oldb, layout=b.fmt)
        else:
            res = res.reshape(olda + oldb)
    else:
        res = res.reshape(olda + oldb)
    return res