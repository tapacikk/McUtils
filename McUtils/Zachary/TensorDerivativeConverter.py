
import abc, numpy as np

__all__ = [
    'TensorDerivativeConverter',
    'TensorExpansionTerms'
]

class TensorExpansionTerms:
    """
    A friend of DumbTensor which exists
    to not only make the tensor algebra suck less but also
    to make it automated by making use of some simple rules
    for expressing derivatives specifically in the context of
    doing the coordinate transformations we need to do.
    Everything here is 1 indexed since that's how I did the OG math
    """
    def __init__(self, qx_terms, xv_terms, qxv_terms=None):
        """
        :param qx_terms:
        :type qx_terms: Iterable[np.ndarray]
        :param xv_terms:
        :type xv_terms: Iterable[np.ndarray]
        """
        self.qx_terms = qx_terms
        self.xv_terms = xv_terms
        self.qxv_terms = qxv_terms
    def QX(self, n):
        return self.QXTerm(self, n)
    def XV(self, m):
        return self.XVTerm(self, m)
    def QXV(self, n, m):
        return self.QXVTerm(self,n,  m)

    class TensorExpansionTerm(metaclass=abc.ABCMeta):
        def __init__(self, array=None):
            self._arr = array
        @abc.abstractmethod
        def dQ(self)->'TensorExpansionTerms.TensorExpansionTerm':
            raise NotImplementedError("base class")
        @abc.abstractmethod
        def asarray(self)->np.ndarray:
            raise NotImplementedError("base class")
        @property
        def array(self):
            if self._arr is None:
                self._arr = self.asarray()
            return self._arr
        @array.setter
        def array(self, arr):
            self._arr = arr
        @abc.abstractmethod
        def rank(self)->int:
            raise NotImplementedError("base class")
        @property
        def ndim(self):
            return self.rank()
        @abc.abstractmethod
        def __repr__(self)->str:
            raise NotImplementedError("base class")
        @abc.abstractmethod
        def simplify(self)->'TensorExpansionTerms.TensorExpansionTerm':
            raise NotImplementedError("base class")
        def __add__(self, other):
            return TensorExpansionTerms.SumTerm(self, other)
        def dot(self, other, i, j):
            return TensorExpansionTerms.ContractionTerm(self, i, j, other)
        def shift(self, i, j):
            return TensorExpansionTerms.AxisShiftTerm(self, i, j)
        def __hash__(self):
            return hash(str(self))
        def __eq__(self, other):
            return str(self) == str(other)
    class SumTerm(TensorExpansionTerm):
        def __init__(self, *terms:'TensorExpansionTerms.TensorExpansionTerm', array=None):
            super().__init__(array=array)
            self.terms=terms
        def dQ(self):
            return type(self)(*(s.dQ() for s in self.terms))
        def asarray(self):
            all_terms = [(i, s.array) for i,s in enumerate(self.terms)]
            clean_terms = [s for s in all_terms if not (isinstance(s[1], (int, float, np.integer, np.floating)) and s[1]==0)]
            for s in clean_terms:
                if s[1].shape != clean_terms[0][1].shape:
                    raise Exception("this term is bad {} in {} (shape {} instead of shape {})".format(self.terms[s[0]], self,
                                                                                                      s[1].shape,
                                                                                                      clean_terms[0][1].shape
                                                                                                      ))
            clean_terms = [c[1] for c in clean_terms]
            try:
                return np.sum(clean_terms, axis=0)
            except:
                raise Exception(clean_terms)
        def rank(self):
            return self.terms[0].ndim
        def simplify(self):
            full_terms = []
            cls = type(self)
            for t in self.terms:
                t = t.simplify()
                if isinstance(t, cls):
                    full_terms.extend(t.terms)
                else:
                    full_terms.append(t)
            full_terms = list(sorted(full_terms, key=lambda t:str(t))) # canonical sorting is string sorting b.c. why not
            return type(self)(*full_terms, array=self._arr)
        def __repr__(self):
            return '+'.join(str(x) for x in self.terms)
        def substitute(self, other):
            """substitutes other in to the sum by matching up all necessary terms"""
            if isinstance(other, TensorExpansionTerms.SumTerm):
                my_terms = list(self.terms)
                for t in other.terms:
                    i = my_terms.index(t)
                    my_terms = my_terms[:i] + my_terms[i+1:]
                my_terms.append(other)
                return type(self)(*my_terms)
            else:
                my_terms = list(self.terms)
                i = my_terms.index(other)
                my_terms = my_terms[:i] + my_terms[i + 1:]
                my_terms.append(other)
                return type(self)(*my_terms)
    class AxisShiftTerm(TensorExpansionTerm):
        def __init__(self, term:'TensorExpansionTerms.TensorExpansionTerm', start:int, end:int, array=None):
            super().__init__(array=array)
            self.term=term
            self.a=start
            self.b=end
        def dQ(self):
            return type(self)(self.term.dQ(), self.a+1, self.b+1)
        def asarray(self):
            t = self.term.array
            if isinstance(t, (int, float, np.integer, np.floating)) and t == 0:
                return 0
            else:
                return np.moveaxis(self.term.array, self.a-1, self.b-1)
        def rank(self):
            return self.term.ndim
        def __repr__(self):
            return '[{}|{}->{}]'.format(self.term, self.a, self.b)
        def simplify(self):
            """We simplify over the possible swap classes"""
            cls = type(self)
            simp = self.term.simplify()
            if isinstance(simp, cls):
                sela = self.a
                selb = self.b
                if sela == selb:
                    return simp
                elif sela > selb: # we flip this around
                    new = cls(simp, selb, sela)
                    for i in range(sela-selb-1):
                        new = cls(new, selb, sela)
                    new._arr = self._arr
                    return new.simplify()
                else:
                    otha = simp.a
                    othb = simp.b
                    otht = simp.term.simplify()
                    if othb == sela:
                        if otha == selb: # inverses although we shouldn't get here...
                            if otht._arr is None:
                                otht._arr = self._arr
                            return otht
                        else: # reduces to a single term
                            new_a = otha
                            new_b = selb
                            return cls(otht, new_a, new_b, array=self._arr)
                    elif otha > sela: # we needed to pick a convention...
                        other = cls(otht, sela, selb)
                        new_a = otha - 0 if otha > selb else 1
                        new_b = othb - 0 if othb > selb else 1
                        new = cls(other, new_a, new_b)
                        if new._arr is None:
                            new._arr = self._arr
                        return new
                    elif otha == sela and othb == selb:
                        # check if we have enough iterates to get the identity
                        is_ident=True
                        for i in range(selb-sela):
                            if isinstance(otht, cls) and otht.a == sela and otht.b == selb:
                                otht = otht.term
                            else:
                                is_ident = False
                                break
                        if is_ident:
                            return otht
                        else:
                            return self
                    else:
                        return self
            elif isinstance(simp, TensorExpansionTerms.SumTerm):
                new = TensorExpansionTerms.SumTerm(
                    *(x.shift(self.a, self.b) for x in simp.terms),
                    array=self._arr
                )
                return new.simplify()
            else:
                return self
    class ContractionTerm(TensorExpansionTerm):
        def __init__(self, left:'TensorExpansionTerms.TensorExpansionTerm', i:int, j:int,
                     right:'TensorExpansionTerms.TensorExpansionTerm', array=None):
            super().__init__(array=array)
            self.left=left
            self.i=i
            self.j=j
            self.right=right
        def asarray(self):
            t1 = self.left.array
            t2 = self.right.array
            if isinstance(t1, (int, float, np.integer, np.floating)) and t1 == 0:
                return 0
            elif isinstance(t2, (int, float, np.integer, np.floating)) and t2 == 0:
                return 0
            else:
                return np.tensordot(self.left.array, self.right.array, axes=[self.i-1, self.j-1])
        def rank(self):
            return self.left.ndim + self.right.ndim - 2
        def dQ(self):
            return (
                type(self)(self.left.dQ(), self.i+1, self.j, self.right)
                + type(self)(
                        self.left,
                        self.i,
                        self.j+1,
                        self.right.dQ()
                    ).shift(self.left.ndim, 1)
            )
        def __repr__(self):
            return '<{}:{},{}:{}>'.format(self.left, self.i, self.j, self.right)
        def simplify(self):
            cls = type(self)
            left = self.left.simplify()
            right = self.right.simplify()
            if isinstance(right, TensorExpansionTerms.AxisShiftTerm):
                # I flip the names just so it agrees with my math...
                t = right.term
                i = right.a
                j = right.b
                a = self.i
                b = self.j
                if b < i:
                    new = cls(left, a, b, t).shift(a-2+i, a-2+j)
                elif b < j:
                    new = cls(left, a, b+1, t).shift(a-1+i, a-2+j)
                elif b == j:
                    new = cls(left, a, i, t)
                else:
                    new = cls(left, a, b, t).shift(a-1+i, a-1+j)
            elif isinstance(left, TensorExpansionTerms.AxisShiftTerm):
                # I flip the names just so it agrees with my math...
                t = left.term
                i = left.a
                j = left.b
                a = self.i
                b = self.j
                if a > j:
                    new = cls(t, a, b, right).shift(i, j)
                elif a == j:
                    new = cls(t, i, b, right)
                elif a >= i:
                    new = cls(t, a+1, b, right).shift(i, j-1)
                else:
                    new = cls(t, a, b, right).shift(i-1, j-1)
            elif isinstance(left, TensorExpansionTerms.SumTerm):
                new = TensorExpansionTerms.SumTerm(
                    *(cls(x, self.i, self.j, right) for x in left.terms),
                    array=self._arr
                )
                new = new.simplify()
            elif isinstance(right, TensorExpansionTerms.SumTerm):
                new = TensorExpansionTerms.SumTerm(
                    *(cls(left, self.i, self.j, x) for x in right.terms),
                    array=self._arr
                )
                new = new.simplify()
            # elif isinstance(self.left, TensorExpansionTerms.QXTerm) and isinstance(self.right, TensorExpansionTerms.XVTerm):
            #     new = TensorExpansionTerms.BasicContractionTerm(self.left.terms, self.left.n, self.i, self.j, self.right.m,
            #                                                      array=self._arr)
            else:
                new = cls(self.left.simplify(), self.i, self.j, self.right.simplify(), array=self._arr)

            if new._arr is None:
                new._arr = self._arr

            return new
    class QXTerm(TensorExpansionTerm):
        def __init__(self, terms:'TensorExpansionTerms', n:int, array=None):
            super().__init__(array=array)
            self.terms = terms
            self.n = n
        def dQ(self):
            return type(self)(self.terms, self.n+1)
        def asarray(self):
            return self.terms.qx_terms[self.n-1]
        def rank(self):
            if isinstance(self.array, (int, float, np.integer, np.floating)) and self.array == 0:
                return self.n + 1
            else:
                return self.array.ndim
        def __repr__(self):
            return 'Q[{}]'.format(self.n)
        def simplify(self):
            return self
    class XVTerm(TensorExpansionTerm):
        def __init__(self, terms:'TensorExpansionTerms', m:int, array=None):
            super().__init__(array=array)
            self.terms = terms
            self.m = m
        def dQ(self):
            mixed_terms = self.terms.qxv_terms
            if (
                    mixed_terms is not None
                    and len(mixed_terms) > 0
                    and len(mixed_terms[0]) >= self.m
                    and mixed_terms[0][self.m-1] is not None
            ):
                return self.terms.QXV(1, self.m)
            else:
                return self.terms.ContractionTerm(self.terms.QX(1), 2, 1, self.terms.XV(self.m+1))
        def asarray(self):
            return self.terms.xv_terms[self.m-1]
        def rank(self):
            if isinstance(self.array, (int, float, np.integer, np.floating)) and self.array == 0:
                return self.m
            else:
                return self.array.ndim
        def __repr__(self):
            return 'V[{}]'.format(self.m)
        def simplify(self):
            return self
    class QXVTerm(TensorExpansionTerm):
        def __init__(self, terms:'TensorExpansionTerms', n:int, m:int, array=None):
            super().__init__(array=array)
            self.terms = terms
            self.n = n
            self.m = m
        def dQ(self):
            return type(self)(self.terms, self.n+1, self.m)
        def asarray(self):
            return self.terms.qxv_terms[self.n-1][self.m-1]
        def rank(self):
            if isinstance(self.array, (int, float, np.integer, np.floating)) and self.array == 0:
                return self.n + self.m
            else:
                return self.array.ndim
        def __repr__(self):
            return 'QV[{},{}]'.format(self.n, self.m)
        def simplify(self):
            return self
    class BasicContractionTerm(TensorExpansionTerm):
        """
        Special case tensor contraction term between two elements of the
        tensor expansion terms.
        """
        def __init__(self, terms:'TensorExpansionTerms', n:int, i:int, j:int, m:int, array=None):
            super().__init__(array=array)
            self.terms = terms
            self.n = n
            self.m = m
            self.i = i
            self.j = j
        def dQ(self):
            return self.terms.SumTerm(
                type(self)(
                    self.terms,
                    self.n + 1,
                    self.i + 1,
                    self.j,
                    self.m
                ),
                self.terms.ContractionTerm(
                    self.terms.QX(1),
                    2,
                    self.n,
                    type(self)(
                        self.terms,
                        self.n,
                        self.i,
                        self.j + 1,
                        self.m + 1
                    )
                )
            )
        def asarray(self):
            return np.tensordot(self.terms.qx_terms[self.n-1], self.terms.xv_terms[self.m-1], axes=[self.i, self.j])
        def rank(self):
            return self.terms.qx_terms[self.n-1].ndim + self.terms.xv_terms[self.m-1].ndim - 2
        def __repr__(self):
            return '<Q[{}]:{},{}:V[{}]>'.format(self.n, self.i, self.j, self.m)
        def simplify(self):
            return self

class TensorDerivativeConverter:
    """
    A class that makes it possible to convert expressions
    involving derivatives in one coordinate system in another
    """
    #TODO: add way to not recompute terms over and over
    def __init__(self, jacobians, derivatives=None, mixed_terms=None):
        """

        :param jacobians: The Jacobian and higher-order derivatives between the coordinate systems
        :type jacobians: Iterable[np.ndarray]
        :param derivatives: Derivatives of some quantity in the original coordinate system
        :type derivatives: Iterable[np.ndarray]
        :param mixed_terms: Mixed derivatives of some quantity involving the new and old coordinates
        :type mixed_terms: Iterable[Iterable[None | np.ndarray]]
        """

        if derivatives is None:
            derivatives = [0] * len(jacobians)

        self.terms = TensorExpansionTerms(jacobians, derivatives, qxv_terms=mixed_terms)

    def convert(self, order=None):

        if order is None:
            order = len(self.terms.qx_terms)

        if order < 0:
            raise ValueError("cannot convert derivatives between coordinates systems to order 0")

        arrays = []
        deriv = self.terms.QX(1).dot(self.terms.XV(1), 2, 1)
        arrays.append(deriv.array)

        for i in range(2, order+1):
            deriv = deriv.dQ()#.simplify() # there's an occasional issue with shift simplifications
            arrays.append(deriv.array)

        return arrays

