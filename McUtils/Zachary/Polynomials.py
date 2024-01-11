import abc, numpy as np, scipy.signal
import itertools as it
from ..Combinatorics import UniquePermutations

__all__ = [
    "AbstractPolynomial",
    "DensePolynomial",
    "SparsePolynomial",
    "PureMonicPolynomial",
    "TensorCoefficientPoly"
]

class AbstractPolynomial(metaclass=abc.ABCMeta):
    """
    Provides the general interface an abstract polynomial needs ot support, including
    multiplication, addition, shifting, access of coefficients, and evaluation
    """

    @property
    @abc.abstractmethod
    def scaling(self):
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def __mul__(self, other)->'AbstractPolynomial':
        raise NotImplementedError("abstract base class")
    @abc.abstractmethod
    def __add__(self, other)->'AbstractPolynomial':
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def shift(self, shift)->'AbstractPolynomial':
        raise NotImplementedError("abstract base class")

    def __rmul__(self, other)->'AbstractPolynomial':
        return self * other
    def __radd__(self, other)->'AbstractPolynomial':
        return self + other
    def __truediv__(self, other)->'AbstractPolynomial':
        return self * (1/other)
    def __neg__(self):
        return -1*self
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return (-self)+other


    # def __rtruediv__(self, other):
    #     return other * (1/self)


class DensePolynomial(AbstractPolynomial):
    """
    A straightforward dense n-dimensional polynomial data structure with
    multiplications and shifts
    """
    def __init__(self,
                 coeffs,
                 prefactor=None,
                 shift=None,
                 stack_dim=0
                 ):
        self._scaling = prefactor
        self._shift = shift
        self._poly_coeffs = np.asanyarray(coeffs)
        self._coeff_tensors = None
        self._uc_tensors = None
        self.stack_dim = stack_dim
    def __repr__(self):
        return "{}({}, {})".format(type(self).__name__, self.coeffs.shape, self.scaling)
    @classmethod
    def from_tensors(cls, tensors, prefactor=None, shift=None, rescale=True):
        coeffs, stack_dim = cls.condense_tensors(tensors, rescale=rescale)
        new = cls(coeffs, prefactor=prefactor, shift=shift, stack_dim=stack_dim)
        return new

    @property
    def shape(self):
        return self._poly_coeffs.shape

    @property
    def scaling(self):
        return 1 if self._scaling is None else self._scaling
    @scaling.setter
    def scaling(self, s):
        self._scaling = s

    @property
    def coeffs(self):
        if self._shift is not None:
            self._poly_coeffs = self._compute_shifted_coeffs(self._poly_coeffs, self._shift, stack_dim=self.stack_dim)
            self._shift = None
        if self._scaling is not None:
            self._poly_coeffs = self._poly_coeffs * self._scaling
            self.scaling = None
        return self._poly_coeffs
    @coeffs.setter
    def coeffs(self, cs):
        self._poly_coeffs = cs

    @property
    def coordinate_dim(self):
        return self._poly_coeffs.ndim - self.stack_dim

    @classmethod
    def _manage_stack_bcast(cls, c1, c2, s1, s2):
        if s1 == s2:
            if s1 == 0:
                return c1, c2, 0
            sd = s1
            final_shape = (np.zeros(c1.shape[:s1]) + np.zeros(c2.shape[:s2])).shape # make numpy broadcast for me
            c1 = np.broadcast_to(c1, final_shape + c1.shape[s1:])
            c2 = np.broadcast_to(c2, final_shape + c2.shape[s2:])
        elif s1 == 0:
            sd = s2
            c1 = np.broadcast_to(np.expand_dims(c1, list(range(s2))))
        elif s2 == 0:
            sd = s1
            c2 = np.broadcast_to(np.expand_dims(c2, list(range(s1))))
        else:
            raise ValueError("can't broadcast shapes {} and {}".format(
                c1.shape[:s1],
                c2.shape[:s2]
            ))
        return c1, c2, sd
    def __mul__(self, other)->'DensePolynomial':
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 1:
                return self
            elif other == 0:
                return 0
        if isinstance(other, DensePolynomial):
            # print(self.coeffs.shape, other.coeffs.shape)
            scs, ocs, sd = self._manage_stack_bcast(
                self.coeffs, other.coeffs,
                self.stack_dim, other.stack_dim
            )
            if sd > 0:
                new_coeffs = np.array([
                    scipy.signal.convolve(s, o)
                    for s,o in zip(
                        scs.reshape((-1,) + scs.shape[sd:]),
                        ocs.reshape((-1,) + ocs.shape[sd:]),
                    )
                ])
                new_coeffs = new_coeffs.reshape(scs.shape[:sd] + new_coeffs.shape[1:])
            else:
                new_coeffs = scipy.signal.convolve(scs, ocs)
            return DensePolynomial(
                new_coeffs,
                shift=None,
                prefactor=self.scaling * other.scaling,
                stack_dim=sd
            )
        else:
            return DensePolynomial(
                self._poly_coeffs,
                shift=self._shift,
                prefactor=self.scaling * other
            )

    def __add__(self, other)->'DensePolynomial':
        if isinstance(other, DensePolynomial):
            stack_dim = max([self.stack_dim, other.stack_dim])

            if self.shape[:self.stack_dim] != other.shape[other.stack_dim]:
                bcast_shape = (np.zeros(self.shape[:self.stack_dim]) * np.zeros(other.shape[:other.stack_dim])).shape
                fcs = np.broadcast_to(self.coeffs, bcast_shape + self.shape[self.stack_dim:])
                ocs = np.broadcast_to(other.coeffs, bcast_shape + other.shape[other.stack_dim:])
            else:
                fcs = self.coeffs
                ocs = other.coeffs

            if self.shape[self.stack_dim:] != other.shape[other.stack_dim:]:  # need to cast to consistent shape
                consistent_shape = [
                    max(s1, s2)
                    for s1, s2 in zip(
                        fcs.shape,
                        ocs.shape
                    )
                ]
                fcs = np.pad(fcs,
                             [[0, c-s] for c,s in zip(consistent_shape, fcs.shape)]
                             )
                ocs = np.pad(other.coeffs,
                             [[0, c-s] for c,s in zip(consistent_shape, ocs.shape)]
                             )

            return DensePolynomial(
                fcs*self.scaling + ocs*other.scaling,
                shift=None,
                stack_dim=stack_dim
            )
        else:
            if self.scaling != 1:
                other = other / self.scaling
            new = self._poly_coeffs.copy().flatten()
            new[0] += other
            new = new.reshape(self._poly_coeffs.shape)
            return DensePolynomial(
                new,
                shift=self._shift,
                prefactor=self.scaling,
                stack_dim=self.stack_dim
            )

    def shift(self, shift)->'DensePolynomial':
        if not isinstance(shift, (int, np.integer, float, np.floating)):
            shift = np.asanyarray(shift)
        return DensePolynomial(
            self._poly_coeffs,
            shift=(0 if self._shift is None else self._shift) + shift,
            prefactor=self._scaling,
            stack_dim=self.stack_dim
        )
    @classmethod
    def _compute_shifted_coeffs(cls, poly_coeffs, shift, stack_dim=0):
        #TODO: make this consistent with non-zero stack_dim


        # if fourier_coeffs is None:
        #     raise ValueError("need fourier coeffs for shifted coeff calc")
        if isinstance(shift, (int, float, np.integer, np.floating)):
            shift = [shift]
        shift = np.asanyarray(shift)

        stack_shape = poly_coeffs.shape[:stack_dim]
        poly_shape = poly_coeffs.shape[stack_dim:]
        poly_dim = len(poly_shape)

        # handle broadcasting of shift
        if stack_dim > 0 and shift.ndim == 1 and shift.shape[0] == poly_dim:
            shift = np.expand_dims(shift, [0]*stack_dim)
        shift = np.broadcast_to(shift, stack_shape + (poly_dim,))

        # factorial outer product
        factorial_terms = np.array([np.math.factorial(x) for x in range(poly_shape[0])])
        for s in poly_shape[1:]:
            factorial_terms = np.expand_dims(factorial_terms, -1) * np.reshape(
                np.array([np.math.factorial(x) for x in range(s)]),
                [1]*factorial_terms.ndim + [s]
            )
        if stack_dim > 0:
            factorial_terms = np.broadcast_to(
                np.expand_dims(factorial_terms, list(range(0, stack_dim))),
                poly_coeffs.shape
            )

        # shift outer product
        if stack_dim > 0:
            shift = np.moveaxis(shift, -1, 0)
            shift_terms = np.power(
                shift[0][..., np.newaxis],
                np.expand_dims(np.arange(poly_shape[0]), list(range(0, stack_dim)))
            )
            for f,s in zip(shift[1:], poly_shape[1:]):
                shift_terms = np.expand_dims(shift_terms, -1) * np.reshape(
                    np.power(f[..., np.newaxis], np.expand_dims(np.arange(s), list(range(0, stack_dim)))),
                    stack_shape + (1,)*(shift_terms.ndim - stack_dim) + (s,)
                )
                # print("-->", shift_terms.shape)
        else:
            shift_terms = np.power(shift[0], np.arange(poly_shape[0]))
            for f, s in zip(shift[1:], poly_shape[1:]):
                shift_terms = np.expand_dims(shift_terms, -1) * np.reshape(
                    np.power(f, np.arange(s)),
                    (1,) * shift_terms.ndim + (s,)
                )

        # print("-->", shift_terms)

        # build shifted polynomial coefficients
        rev_fac = factorial_terms
        for i in range(factorial_terms.ndim - stack_dim):
            rev_fac = np.flip(rev_fac, axis=stack_dim+i)
            shift_terms = np.flip(shift_terms, axis=stack_dim+i)

        poly_terms = poly_coeffs * factorial_terms
        shift_terms = shift_terms / rev_fac

        # print(poly_coeffs.shape, factorial_terms.shape, shift_terms.shape)
        if stack_dim > 0:
            new = np.array([
                scipy.signal.convolve(p, s)
                for p,s in zip(
                    poly_terms.reshape((-1,)+poly_shape),
                    shift_terms.reshape((-1,)+poly_shape)
                )
            ])
            new = new.reshape(stack_shape + new.shape[1:])
        else:
            new = scipy.signal.convolve(
                poly_terms,
                shift_terms
            )

        # stack_slice = (slice(None),)*stack_dim
        for i,s in enumerate(poly_shape):
            new = np.take(new, np.arange(-s, 0), axis=stack_dim+i)
        new = new / factorial_terms

        return new

    @classmethod
    def extract_tensors(cls, coeffs, stack_dim=None, permute=True, rescale=True):
        if stack_dim is None:
            stack_dim = 0
            stack_shape = ()
        else:
            stack_shape = coeffs.shape[:stack_dim]

        nz_pos = np.where(coeffs != 0)
        if len(nz_pos) == 0 or len(nz_pos[0]) == 0:
            return [0]
        max_order = np.max(np.sum(nz_pos[stack_dim:], axis=0, dtype=int))
        # raise Exception(max_order)
        # max_order = sum(c-1 for c in coeffs.shape)
        num_dims = len(coeffs.shape) - stack_dim
        tensors = [
            np.zeros(stack_shape + (num_dims,)*i)
            for i in range(1, max_order+1)
        ]
        ref = [0] if stack_dim == 0 else [np.zeros(stack_shape)]
        tensors = ref + tensors # include ref
        for idx in zip(*nz_pos): # TODO: speed this up...
            # we compute the order of the index, compute the number of
            # permutations of the index, and then set all of the permutations
            # equal to the value divided by the number of perms
            order = sum(idx[stack_dim:])
            value = coeffs[idx]
            if order == 0:
                if stack_dim > 0: # this corresponds to an element of only one of the polynomials
                    tensors[0][idx[:stack_dim]] = value
                else:
                    tensors[0] = value
            else:
                new_idx = sum(
                    ([i]*n for i,n in enumerate(idx[stack_dim:])),
                    []
                ) # we figure out which coords are affected and by how much
                if permute:
                    final_perms = UniquePermutations(new_idx).permutations()
                else:
                    final_perms = np.array([new_idx])
                # print(final_perms)
                if rescale:
                    subvalue = value / len(final_perms)
                else:
                    subvalue = value
                fp_inds = tuple(final_perms.T)
                if stack_dim > 0: # this corresponds to an element of only one of the polynomials
                    extra_idx = idx[:stack_dim]
                    nels = len(fp_inds[0])
                    bcast_idx = tuple(np.full(nels, i, dtype=int) for i in extra_idx)
                    fp_inds = bcast_idx + fp_inds
                tensors[order][fp_inds] = subvalue
        return tensors
    @classmethod
    def condense_tensors(cls, tensors, rescale=True):
        # we'll be a bit slower here in constructing so we can make sure
        # we get the smallest tensor

        if not isinstance(tensors[0], (int, float, np.integer, np.floating)):
            tensors = [np.asanyarray(t) for t in tensors]
            stack_shape = np.asanyarray(tensors[0]).shape
        else:
            tensors = [tensors[0]] + [np.asanyarray(t) for t in tensors[1:]]
            stack_shape = ()
        stack_dim = len(stack_shape)

        # order = len(tensors)
        if len(tensors) == 1:
            raise NotImplementedError("constant polynomial support coming")

        ncoord = tensors[1].shape[-1]
        tensor_elems = {}
        shape = [1] * ncoord
        # we could be more efficient through where tricks if we wanted
        for o,t in enumerate(tensors):
            if o == 0:
                for idx in zip(*np.where(t != 0)):
                    tidx = idx + (0,)*ncoord
                    tensor_elems[tidx] = t[idx]
            else:
                utri_pos = tuple(np.array(list(it.combinations_with_replacement(range(ncoord), r=o))).T)
                if stack_dim > 0:
                    utri_pos = (slice(None),)*stack_dim + utri_pos
                vals = t[utri_pos]
                nz_pos = np.where(vals != 0)
                for p in zip(*nz_pos):
                    val = vals[p]
                    upos = [u[p[-1]] for u in utri_pos[stack_dim:]]
                    idx = tuple(np.bincount(upos, minlength=ncoord))
                    if rescale:
                        val = val * UniquePermutations.count_permutations(idx)
                    tidx = p[:-1] + idx
                    tensor_elems[tidx] = val
                    for n, i in enumerate(idx):
                        shape[n] = max(i + 1, shape[n])

        # raise Exception(shape)
        condensed_tensor = np.zeros(list(stack_shape) + shape)
        for idx,val in tensor_elems.items():
            condensed_tensor[idx] = val
        return condensed_tensor, stack_dim

    @property
    def coefficient_tensors(self):
        if self._coeff_tensors is None:
            self._coeff_tensors = self.extract_tensors(self.coeffs, stack_dim=self.stack_dim)
        return self._coeff_tensors
    @property
    def unscaled_coefficient_tensors(self):
        if self._uc_tensors is None:
            self._uc_tensors = self.extract_tensors(self.coeffs, rescale=False, stack_dim=self.stack_dim)
        return self._uc_tensors
    def transform(self, lin_transf):
        """
        Applies (for now) a linear transformation to the polynomial
        """
        #TODO: make this consistent with non-zero stack_dim
        if self.stack_dim != 0:
            raise NotImplementedError("need to make work with non-zero stack dims")

        tensors = self.coefficient_tensors
        transf_tensors = []
        for n,t in enumerate(tensors):
            for ax in range(n):
                t = np.tensordot(lin_transf, t, axes=[1, n-1])
            transf_tensors.append(t)
        new = self.condense_tensors(transf_tensors)
        return DensePolynomial(new)

    def outer(self, other):
        if self.stack_dim != 0 or other.stack_dim != 0:
            raise NotImplementedError("need to make work with non-zero stack dims")
        return DensePolynomial(
            np.multiply.outer(self.coeffs, other),
            prefactor=self.scaling*other.scaling
        )

    @classmethod
    def _coord_deriv(cls, coeffs, coord, stack_dim):
        coord = coord + stack_dim
        if coeffs.shape[coord] == 1:
            return 0
        n = coeffs.shape[coord]
        scaling = np.arange(1, n)
        scaling = np.expand_dims(scaling, list(range(coord)) + list(range(coord+1, coeffs.ndim)))
        new_coeffs = scaling * np.take(coeffs, np.arange(1, n), axis=coord)
        return new_coeffs
    def deriv(self, coord):
        if self._poly_coeffs.shape[self.stack_dim + coord] == 1:
            return 0
        new_coeffs = self._coord_deriv(self.coeffs, coord, self.stack_dim)
        return DensePolynomial(
            new_coeffs,
            prefactor=self.scaling,
            stack_dim=self.stack_dim
        )

    @classmethod
    def _apply_grad(cls, coeffs, stack_dim):
        shape = coeffs.shape
        n = coeffs.ndim - stack_dim
        c = shape[stack_dim:]

        new_shape = shape[:stack_dim] + (n,) + c
        new_coeffs = np.zeros(new_shape)
        for i in range(n):
            sub_coeffs = cls._coord_deriv(coeffs, i, stack_dim)
            if not isinstance(sub_coeffs, np.ndarray):
                if sub_coeffs != 0: raise ValueError("...?")
                continue
            # construct slice spec for inserting into new coeffs
            insert_spec = (
                    # [i, :, ..., :, :-1, :, ..., :]
                    (slice(None),) * stack_dim
                    + (i,)
                    + (slice(None),) * i
                    + (slice(None, c[i] - 1),)
                    + (slice(None),) * (n - (i + 1))
            )
            new_coeffs[insert_spec] = sub_coeffs
        return new_coeffs

    def grad(self):

        return DensePolynomial(
            self._apply_grad(self.coeffs, self.stack_dim),
            prefactor=self._scaling,
            stack_dim=self.stack_dim+1
        )


    def clip(self, threshold=1e-15):
        clipped_inds = np.where(np.abs(self.coeffs * self.scaling) > threshold)
        if len(clipped_inds) == 0 or len(clipped_inds[0]) == 0:
            return 0
        new_shape = tuple(max(x) + 1 for x in clipped_inds)
        coeffs = np.zeros(new_shape)
        coeffs[clipped_inds] = self.coeffs[clipped_inds]
        return DensePolynomial(
            coeffs,
            prefactor=self._scaling,
            stack_dim=self.scaling
        )
    
class SparsePolynomial(AbstractPolynomial):
    """
    A semi-symbolic representation of a polynomial of tensor
    coefficients
    """
    def __init__(self, terms:dict, prefactor=1):
        self.terms = terms
        self.prefactor = prefactor
        self._shape = None

    @property
    def scaling(self):
        return 1 if self.prefactor is None else self.prefactor
    @scaling.setter
    def scaling(self, s):
        self.prefactor = s

    def expand(self):
        if self.prefactor == 1:
            return self
        else:
            return type(self)({k:self.prefactor*v for k,v in self.terms.items()}, prefactor=1)
    @classmethod
    def monomial(cls, idx, value=1):
        return cls({idx:value})
    def __repr__(self):
        return "{}({},{})".format(type(self).__name__, self.terms,self.prefactor)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 1:
                return self
            elif other == 0:
                return SparsePolynomial({})
        if isinstance(other, SparsePolynomial):
            new_terms = {}
            for k,v in other.terms.items():
                for k2,v2 in self.terms.items():
                    t = tuple(sorted(k + k2))
                    new_terms[t] = new_terms.get(t, 0) + v * v2
                    if new_terms[t] == 0:
                        del new_terms[t]
            return type(self)(new_terms, prefactor=self.prefactor*other.prefactor)
        else:
            return type(self)(self.terms, prefactor=self.prefactor*other)
            # new_terms = {}
            # for k,v in self.terms.items():
            #     new_terms[k] = self.prefactor*other*v
            # return SparsePolynomial(new_terms)
    def __add__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0:
                return self
            else:
                return self + type(self)({():other})
        else:
            self = self.expand()
            new_terms = {}
            if isinstance(other, SparsePolynomial):
                other = other.expand()
                for s in other.terms.keys() & self.terms.keys():
                    v = other.terms[s]
                    v2 = self.terms[s]
                    new = v + v2
                    if new != 0:
                        new_terms[s] = new
                for o in other.terms.keys() - self.terms.keys():
                    new_terms[o] = other.terms[o]
                for o in self.terms.keys() - other.terms.keys():
                    new_terms[o] = self.terms[o]
            else:
                for k2, v2 in self.terms.items():
                    new = v2 + other
                    if new != 0:
                        new_terms[k2] = new
            if len(new_terms) == 0:
                return 0
            return type(self)(new_terms)

    @classmethod
    def _to_tensor_idx(cls, term, ndim):
        base = [0]*ndim
        for idx, cnt in zip(*np.unique(term, return_counts=True)): #TODO: kinda dumb to do this 2x but w/e
            base[idx] = cnt
        return tuple(base)
        # idx_terms = {}
        # m = 0
        # for t in idx:
        #     m = max(m, t)
        #     idx_terms[t] = idx_terms.get(t, 0) + 1
        # return tuple(idx_terms[t] for t in range(m))
    @property
    def shape(self):
        if self._shape is None:
            max_dims = {}
            for t in self.terms.keys():
                for idx,cnt in zip(*np.unique(t, return_counts=True)): # can I assume sorted?
                    max_dims[idx] = max(max_dims.get(idx, 1), cnt + 1)
            max_key = max(list(max_dims.keys()))
            self._shape = tuple(max_dims.get(k, 1) for k in range(max_key+1))
        return self._shape
    def as_dense(self)->DensePolynomial:
        new_coeffs = np.zeros(self.shape)
        ndim = new_coeffs.ndim
        for key, val in self.terms.items():
            idx = self._to_tensor_idx(key, ndim)
            new_coeffs[idx] = val
        return DensePolynomial(
            new_coeffs,
            prefactor=self.prefactor
        )
    def shift(self, shift)->DensePolynomial:
        return self.as_dense().shift(shift)

class PureMonicPolynomial(SparsePolynomial):
    def __init__(self, terms: dict, prefactor=1, canonicalize=True):
        if canonicalize:
            terms = {self.canonical_key(k):v for k,v in terms.items()}
        super().__init__(terms, prefactor)

    @property
    def shape(self):
        raise ValueError("{} doesn't have a dense counterpart".format(type(self).__name__))
    def as_dense(self):
        raise ValueError("{} doesn't have a dense counterpart".format(type(self).__name__))
    def shift(self, shift) -> DensePolynomial:
        raise ValueError("{} doesn't have a dense counterpart".format(type(self).__name__))

    @classmethod
    def monomial(cls, idx, value=1):
        return cls({(idx,): value})

    @classmethod
    def key_hash(cls, monomial_tuple):
        # hashes tuples of indices to give a fast check that the tuples
        # are the same independent of ordering
        return sum(hash(x) for x in monomial_tuple)

    @classmethod
    @abc.abstractmethod
    def canonical_key(cls, monomial_tuple):
        raise NotImplementedError("{} is abstract".format(cls.__name__))

    def direct_multiproduct(self, other, key_value_generator):
        if not isinstance(other, PureMonicPolynomial):
            raise TypeError("doesn't make sense")

        new_terms = {}
        term_hashes = {}
        for k, v in other.terms.items():
            for k2, v2 in self.terms.items():
                for k3, v3 in key_value_generator(k, k2, v, v2):
                    new_hash = self.key_hash(k3) + self.key_hash(k2)
                    if new_hash in term_hashes:
                        t = term_hashes[new_hash]
                    else:
                        t = self.canonical_key(k3)  # construct new tuple and canonicalize
                        term_hashes[new_hash] = t
                    new_terms[t] = new_terms.get(t, 0) + v3  # this is the case where after mult. two terms merge
                    if new_terms[t] == 0:
                        del new_terms[t]
        if self.prefactor is None:
            pref = other.prefactor
        elif other.prefactor is None:
            pref = self.prefactor
        else:
            pref = self.prefactor * other.prefactor
        return type(self)(new_terms, prefactor=pref, canonicalize=False)
    def direct_product(self, other, key_func=None, mul=None):
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 1:
                return self
            elif other == 0:
                return 0
                # return type(self)({})
        if isinstance(other, PureMonicPolynomial):
            if mul is None:
                mul = lambda a,b:a*b
            if key_func is None:
                key_func = lambda key1,key2:k+k2
            new_terms = {}
            term_hashes = {}
            for k,v in other.terms.items():
                for k2,v2 in self.terms.items():
                    new_hash = self.key_hash(k) + self.key_hash(k2)
                    if new_hash in term_hashes:
                        t = term_hashes[new_hash]
                    else:
                        t = self.canonical_key(key_func(k, k2)) # construct new tuple and canonicalize
                        term_hashes[new_hash] = t
                    new_terms[t] = new_terms.get(t, 0) + mul(v, v2) # this is the case where after mult. two terms merge
                    if new_terms[t] == 0:
                        del new_terms[t]
            if self.prefactor is None:
                pref = other.prefactor
            elif other.prefactor is None:
                pref = self.prefactor
            else:
                pref = self.prefactor*other.prefactor
            return type(self)(new_terms, prefactor=pref, canonicalize=False)
        else:
            return type(self)(self.terms, prefactor=self.prefactor*other if self.prefactor is not None else other, canonicalize=False)
    def __mul__(self, other):
        return self.direct_product(other)
class TensorCoefficientPoly(PureMonicPolynomial):
    """
    Represents a polynomial constructed using tensor elements as monomials
    by tracking sets of indices
    """

    @classmethod
    def canonical_key(cls, monomial_tuple):
        # we need a way to sort indices, which we do by grouping by key length,
        # doing a standard sort for each length, and reconcatenating
        s_groups = {}
        for index_tuple in monomial_tuple:
            l = len(index_tuple)
            grp = s_groups.get(l, [])
            s_groups[l] = grp
            grp.append(index_tuple)
        t = tuple(
            grp
            for l in sorted(s_groups.keys())
            for grp in sorted(s_groups[l])
        )
        return t

# class TensorCoeffPoly(AbstractPolynomial):
#     """
#     A semi-symbolic representation of a polynomial of tensor
#     coefficients
#     """
#
#     def __init__(self, terms:dict, prefactor=1):
#         self.terms = terms
#         self.prefactor = prefactor
#     def expand(self):
#         if self.prefactor == 1:
#             return self
#         else:
#             return TensorCoeffPoly({k:self.prefactor*v for k,v in self.terms.items()}, prefactor=1)
#     @classmethod
#     def monomial(cls, idx, value=1):
#         return cls({(idx,):value})
#     def __repr__(self):
#         return "{}({},{})".format(type(self).__name__, self.terms,self.prefactor)