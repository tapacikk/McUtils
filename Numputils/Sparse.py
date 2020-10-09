import numpy as np, scipy.sparse as sp, itertools as ip, functools as fp, os

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
        self._block_inds = None # cached to speed things up
        self._block_vals = None # cached to speed things up
    def _init_matrix(self):
        a = self._a
        if isinstance(a, SparseArray):
            if self.fmt is not a.fmt:
                self._a = self.fmt(a._a, shape=a.shape)
            else:
                self._a = a._a
            shp = self._shape
            if shp is not None:
                self.reshape(a.shape)
            else:
                self._shape = a.shape
        elif isinstance(a, sp.spmatrix):
            if self._shape is None:
                self._shape = a.shape
        elif isinstance(a, np.ndarray):
            self._shape = a.shape
            if len(a.shape) > 2:
                a = a.reshape((a.shape[0], np.prod(a.shape[1:])))
            elif len(a.shape) == 1:
                a = a.reshape(a.shape + (1,))
            self._a = self.fmt(a, shape=a.shape)
        # we're gonna support the (vals, (i_1, i_2, i_3, ...)) syntax for constructing
        # an array based on its non-zero positions
        elif len(a) == 2 and len(a[1]) > 0 and len(a[0]) == len(a[1]):
            block_vals, block_inds = a
            block_inds = tuple(np.array(i, dtype=int) for i in block_inds)
            if self._shape is None:
                self._shape = tuple(np.max(x) for x in block_inds)
            if len(block_inds) != len(self._shape):
                raise ValueError("{}: can't initialize array of shape {} from non-zero indices of dimension {}".format(
                    type(self).__name__,
                    self._shape,
                    len(block_inds)
                ))
            # gotta make sure our inds are sorted so we don't run into sorting issues later...
            flat = self._ravel_indices(a, self._shape)
            sort = np.argsort(flat)
            flat = flat[sort]
            block_vals = block_vals[sort]
            block_inds = tuple(i[sort] for i in block_inds)
            total_shape = (1, len(block_vals))
            init_inds = (np.zeros(len(block_vals)), flat)
            try:
                data = self.fmt((block_vals, init_inds))
            except TypeError:
                data = self.fmt(sp.csc_matrix((block_vals, init_inds)))
            self._a = data
            self._block_vals = block_vals
            self._block_inds = block_inds
        else:
            non_sparse, sparse = self._get_shape()
            if non_sparse is None:
                self._a = np.array(self._a)
                self._init_matrix()
            else:
                self._shape = non_sparse + sparse
                data, other = self._get_data(non_sparse, sparse)
                block_data, inds, total_shape = other
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
        """
        Walks through the array data we're holding onto and determines
        where the sparse blocks start
        """
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
    @property
    def non_zero_count(self):
        return self.data.nnz
    # def __len__(self):
    #     return self.shape[0]

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

    _ravel_cache = {}  # hopefully faster than bunches of unravel_index calls...
    @classmethod
    def _ravel_indices(cls, mult, dims):
        # we're hoping that we call with `n` often enough that we get a performance benefit
        if isinstance(dims, list):
            dims = tuple(dims)
        if dims not in cls._ravel_cache:
            cls._ravel_cache[dims] = {}
        cache = cls._ravel_cache[dims]
        if isinstance(mult[0], np.ndarray):
            n_hash = hash(tuple(m.data.tobytes() for m in mult))
        else:
            try:
                n_hash = hash(mult)
            except TypeError:
                # means hashing failed so we just shortcut out rather than try to be clever
                # raise Exception([mult, dims])
                return np.ravel_multi_index(mult, dims)
        if n_hash in cache:
            res = cache[n_hash]
        else:
            res = np.ravel_multi_index(mult, dims)
            cache[n_hash] = res
        return res
    @property
    def block_vals(self):
        if self._block_vals is None:
            d = self.data
            row_inds, col_inds, data = sp.find(d)
            flat = self._ravel_indices((row_inds, col_inds), d.shape)
            unflat = self._unravel_indices(flat, self.shape)
            self._block_inds = (flat, unflat)
            self._block_vals = data
        return self._block_vals
    @property
    def block_inds(self):
        if self._block_inds is None:
            vals = self.block_vals
        return self._block_inds
    @block_inds.setter
    def block_inds(self, bi):
        if isinstance(bi, tuple) and len(bi) == 2:
            flat, unflat = bi
            if isinstance(unflat[0], int):
                row_inds, col_inds = bi
                flat = self._ravel_indices((row_inds, col_inds), self.data.shape)
                unflat = self._unravel_indices(flat, self.shape)
        elif isinstance(bi[0], int):
            flat = bi
            unflat = self._unravel_indices(flat, self.shape)
        else:
            unflat = bi
            flat = self._ravel_indices(bi, self.shape)
        if len(unflat) != len(self.shape):
            raise ValueError("{}: block indices must have same dimension as array ({}); was given {}".format(
                type(self).__name__,
                len(self.shape),
                len(unflat)
            ))
        if self._block_vals is not None and len(flat) != len(self._block_vals):
                raise ValueError("{}: number of block indices must match number of non-zero elements ({}); was given {}".format(
                    type(self).__name__,
                    len(self._block_vals),
                    len(flat)
                ))
        self._block_inds = (flat, unflat)
    @property
    def block_data(self):
        res = self.block_vals, self.block_inds[1]
        return res

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
        new_shape = tuple(shp[i] for i in transp)
        if len(new_shape) > 2:
            total_shape = (np.prod(new_shape[:-2])*new_shape[-2], new_shape[-1])
            flat = self._ravel_indices(new_inds, new_shape)
            unflat = self._unravel_indices(flat, total_shape)
        else:
            flat = None
            unflat = new_inds
            total_shape = new_shape
        data = self._build_data(data, unflat, total_shape)
        new = type(self)(data, shape = new_shape, layout = self.fmt)
        arr = np.lexsort(unflat)
        new_inds = [inds[arr] for inds in new_inds]
        if self._block_vals is not None:
            new_v = self._block_vals[arr]
            new._block_vals = new_v
        if flat is None:
            new.block_inds = new_inds
        else:
            # try:
            new.block_inds = (flat, new_inds)
            # except:
            #     raise Exception(new_shape, len(total_shape))
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
    def __radd__(self, other):
        return self.plus(other)
    def plus(self, other):
        d = self.data
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0.:
                return self.copy()
            else:
                d = other + d
                new = type(self)(d, shape=self.shape, layout=self.fmt)
                if self._block_vals is not None:
                    bvs = self._block_vals
                    bi = self._block_inds
                    new._block_vals = other + bvs
                    new._block_inds = bi
                return new

        if isinstance(other, SparseArray):
            other = other.data
        if isinstance(other, sp.spmatrix):
            other = other.reshape(d.shape)
        elif isinstance(other, np.ndarray):
            other = np.broadcast_to(other, self.shape).reshape(d.shape)
        new = d.__add__(other)
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
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0.:
                new = self.fmt(d.shape)
                return type(self)(new, shape=self.shape, layout=self.fmt)
            elif other == 1.:
                return self.copy()
            else:
                d = other * self.data
                new = type(self)(d, shape=self.shape, layout=self.fmt)
                if self._block_vals is not None:
                    bvs = self._block_vals
                    bi = self._block_inds
                    new._block_vals = other * bvs
                    new._block_inds = bi
                return new

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

    def copy(self):
        import copy
        return copy.copy(self)

    def _get_element(self, idx):
        """
        Convert idx into a 1D index or slice or whatever and then convert it back to the appropriate 2D shape

        :param i:
        :type i:
        :return:
        :rtype:
        """

        # we check first to see if we were asked for just a single vector of elements
        # the detection heuristic is basically: is everything just a slice of ints or nah
        if isinstance(idx, (int, np.integer)):
            idx = (idx,)
        pull_elements = len(idx) == len(self.shape) and all(isinstance(x, (int, np.integer)) for x in idx)
        if not pull_elements:
            pull_elements = all(not isinstance(x, (int, np.integer, slice)) for x in idx)
            if pull_elements:
                e1 = len(idx[0])
                pull_elements = all(len(x) == e1 for x in idx)

        if pull_elements:
            flat = self._ravel_indices(idx, self.shape)
            unflat = self._unravel_indices(flat, self.data.shape)
            res = self.data[unflat]
            if not isinstance(flat, int):
                res = np.array(res)
            return res
        else:
            # need to compute the shape of the resultant block
            blocks = [
                (
                    np.array([i]) if isinstance(i, (int, np.integer)) else (
                        np.arange(s)[i,].flatten()
                    )
                )
                for i, s in zip(idx, self.shape)
            ]
            # we filter out places where new_shape[i] == 1 at a later stage
            # for now we just build out the total shape it _would_ have with axes of len 1
            new_shape = [len(x) for x in blocks] + list(self.shape[len(blocks):])

            # now we iterate over each block and use it as a successive filter on our non-zero positions
            data, inds = self.block_data
            inds = list(inds)

            def g(b, j):
                """
                finds the positions where the block & index align
                """
                w = np.argwhere(b == j)
                if len(w) > 0:
                    w = w[0][0]
                else:
                    w = -1
                return w
            for i, b, s in zip(range(len(blocks)), blocks, self.shape):
                k = 0
                ixs = inds[i]
                # we add up the indices to give a list of 0 & 1 to use as a mask
                # we use sum because sum is boolean OR
                # this will give us the elements of ixs where
                filter = np.sum(ixs == j for j in b).astype(bool)
                # we then apply this to the non-zero indices and values we're tracking
                inds = [ix[filter] for ix in inds]
                data = data[filter]
                # finally, we remap the current set of indices so that indices that are
                # disappearing get removed and the ones that are staying get shifted down
                # to match that change
                mapping = np.array([g(b, j) for j in range(s)])
                inds[i] = mapping[inds[i]]

            # now that we've filtered our data, we filter out axes of size 1
            # print(inds, new_shape)
            inds = [ix for ix, j in zip(inds, new_shape) if j > 1]
            new_shape = tuple(j for j in new_shape if j > 1)
            # we also apply this to the indices that we're filtering on

            # if idx[0] == 0 and isinstance(idx[2], slice):
            #     raise Exception(blocks)

            # finally, we track our indices so that we don't need to recompute anything later
            if len(new_shape) > 2:
                total_shape = (np.prod(new_shape[:-2]) * new_shape[-2], new_shape[-1])
                flat = self._ravel_indices(inds, new_shape)
                unflat = self._unravel_indices(flat, total_shape)
            elif len(new_shape) == 1:
                flat = None
                unflat = inds+[np.zeros((len(inds[0]),))]
                total_shape = new_shape+(1,)
            else:
                flat = None
                unflat = inds
                total_shape = new_shape

            # raise Exception(blocks, new_shape, len(inds), len(unflat))
            od = data
            try:
                data = self._build_data(data, unflat, total_shape)
            except Exception as e:
                # print(data.shape, unflat)
                data = e
            # raise Exception(unflat, od, data, inds, total_shape)
            if isinstance(data, Exception):
                raise IndexError("{}: couldn't take element {} of array {} (Got Error: '{}')".format(
                    type(self).__name__,
                    idx,
                    self,
                    data
                ))

            new = type(self)(data, shape=new_shape, layout=self.fmt)
            if flat is None:
                new.block_inds = inds
            else:
                new.block_inds = flat, inds
            return new
    def savez(self, file, compressed=True):
        """
        Saves a SparseArray to a file (must have the npz extension)
        :param file:
        :type file:
        :param compressed:
        :type compressed:
        :return: the saved file
        :rtype: str
        """

        # sp.save_npz already sticks the extension on so it doesn't hurt to explicitly add it...
        if isinstance(file, str) and os.path.splitext(file)[1] != ".npz":
            file += ".npz"
        sp.save_npz(file, self.data, compressed=compressed)
        bleh = np.load(file)
        flat, unflat = self.block_inds
        bv = self.block_vals
        if compressed:
            np.savez_compressed(
                file,
                _block_shape=np.array(self.shape),
                _block_inds_flat=np.array(flat),
                _block_inds_unflat=np.array(unflat),
                _block_vals=bv,
                compressed=compressed,
                **bleh
            )
        else:
            np.savez(
                file,
                _block_shape=np.array(self.shape),
                _block_inds_flat=np.array(flat),
                _block_inds_unflat=np.array(unflat),
                _block_vals=bv,
                compressed=compressed,
                **bleh
                )
        return file
    @classmethod
    def loadz(cls, file):
        """
        Loads a SparseArray from an npz file
        :param file:
        :type file:
        :return:
        :rtype: SparseArray
        """
        data = sp.load_npz(file) #type: sp.spmatrix
        other = np.load(file)
        new = cls(data, shape=other['_block_shape'], layout=type(data), initialize=False)
        new._block_inds = (other['_block_inds_flat'], other['_block_inds_unflat'])
        new._block_inds = other['_block_vals']
        return new

    def __getitem__(self, item):
        return self._get_element(item)

    def __repr__(self):
        return "{}(<{}> nonzero={})".format(type(self).__name__,
                                           ", ".join([str(x) for x in self.shape]),
                                          self.non_zero_count
                                           )

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