"""
LazyTensors provides a small framework for symbolically working with Tensors
"""

import numpy as np
from abc import *

########################################################################################################################
#
#                                           Tensor
#
class Tensor:
    """A semi-symbolic representation of a tensor. Allows for lazy processing of tensor operations."""
    def __init__(self, a, shape = None):
        self._a = a
        if shape is None:
            shape = a.shape
        self.shape = shape
    @classmethod
    def from_array(cls, a, shape = None):
        if isinstance(a, Tensor):
            return a
        else:
            cls(a, shape = shape)
    @property
    def array(self):
        if isinstance(self._a, np.ndarray):
            return self._a
        else:
            return self._a.array
    def get_dim(self):
        return len(self.shape)
    @property
    def dim(self):
        return self.get_dim()
    def __mul__(self, other):
        if isinstance(other, float):
            return TensorMul(self, other)
        else:
            return TensorDot(self, other)
    def __rmul__(self, other):
        if isinstance(other, float):
            return TensorMul(self, other)
        else:
            return TensorDot(self, other)
    def __add__(self, other):
        return TensorPlus(self, other)
    def __pow__(self, power, modulo=None):
        return TensorPow(self, power)

    def handle_missing_indices(self, missing, extant):
        # the default assumption is basically to use the diagonal element -- i.e. to assume that only the extant indices matter
        return self._a[extant]
    def pull_index(self, *idx):
        dim = len(idx)
        sdim = self.dim
        if dim > sdim:
            raise ValueError("{} has only {} dimensions but {} were requested".format(self, sdim, dim))
        else:
            real_shape = self._a.shape
            # here we add our hacks
            real_dim = len(real_shape)
            if real_dim < dim:
                a = self.handle_missing_indices(idx[:dim-real_dim], idx[dim-real_dim:])
            else:
                a = self._a[idx]
            try:
                new = type(self)(a)
            except ValueError:
                new = Tensor(a)
            return new

    def __getitem__(self, item):
        self.pull_index(*item)

    def __repr__(self):
        return "{}(<{}>)".format(type(self).__name__, ", ".join(*(str(x) for x in self.shape)))

########################################################################################################################
#
#                                           TensorOps
#
class TensorOp(Tensor):
    """A lazy representation of tensor operations to save memory"""
    def __init__(self, a, b):
        # if isinstance(a, Tensor):
        #     a = a._a
        # if isinstance(b, Tensor):
        #     a = b._a
        self._a = a
        self._b = b

    def op(self, a, b):
        raise NotImplementedError(
            "{}: {} not implemented until a subclass does it".format(type(self.__name__), 'op')
        )
    def get_shape(self, a, b):
        return a.shape
    @property
    def shape(self):
        return self.get_shape(self._a, self._b)
    @property
    def array(self):
        """Ought to always compile down to a proper ndarray

        :return:
        :rtype: np.ndarray
        """
        return self.op(self._a, self._b)
    def __getitem__(self, i):
        return self.op(self._a[i], self._b)
class TensorPlus(TensorOp):
    """Represents an addition of two tensors"""
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array # with addition we can't really avoid it...
        if isinstance(b, Tensor):
            b = b.array
        return a + b
    def get_shape(self, a, b):
        return super().get_shape(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b[i])
class TensorPow(TensorOp):
    """Represents an addition of two tensors"""
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array
        return np.linalg.matrix_power(a, b)
    def get_shape(self, a, b):
        return super().get_shape(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b)
class TensorMul(TensorOp):
    """Represents a multiplication of a tensor and a scalar"""
    def op(self, a, b):
        if isinstance(a, Tensor):
            a = a.array # with scalar mult we can't really avoid it...
        return a * b
    def get_shape(self, a, b):
        return super().get_shape(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b)
class TensorDot(TensorOp):
    """Represents a tensor contraction across the main axis only"""
    def get_shape(self, a, b):
        return a.shape[:-1] + b.shape[1:]
    def op(self, a, b):
        # we'll allow this to cast to dense as we'll assume people are only ever calling this on low dimensional tensors...
        if isinstance(a, Tensor):
            a = a.array
        if isinstance(b, Tensor):
            b = b.array
        return np.tensordot(a, b)
    def __getitem__(self, i):
        return type(self)(self._a[i], self._b)
########################################################################################################################
#
#                                           TensorDerivative
#
class TensorDerivative(Tensor):
    def __init__(self, tensor):
        ...

########################################################################################################################
#
#                                           LazyOperatorTensor
#
class LazyOperatorTensor(Tensor):
    """A super-lazy tensor that represents the elements of an operator """

    def __init__(self, operator, shape, memoization = True):
        if memoization is True:
            memoization = {}
        self.memoization = isinstance(memoization, dict)
        if self.memoization:
            a = memoization
        else:
            a = None

        self.operator = operator
        super().__init__(a, shape=shape)

    @property
    def array(self):
        import itertools as it

        base = np.full(self.shape, None, dtype=object)
        mem = self._a
        try:
            self._a = None
            _cast = False
            for idx in it.product(range(x) for x in self.shape):
                res = self._get_element(idx)
                if not _cast:
                    if isinstance(res, (int, float, np.integer, np.float)):
                        base = np.zeros(base, dtype=type(res))
                    _cast = True

                base[idx] = res
        finally:
            self._a = mem

        return base

    def _get_element(self, indices):
        if self._a is not None:
            try:
                res = self._a[indices]
            except (KeyError, IndexError):
                res = self.operator(indices)
                self._a[indices] = res
            return res
        else:
            return self.operator(indices)

    def __getitem__(self, item):
        return self._get_element(item)
