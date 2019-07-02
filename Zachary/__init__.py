"""
A small framework for handling finite differencing of functions
"""

from .FiniteDifferenceFunction import FiniteDifferenceFunction, finite_difference
from .Derivatives import FiniteDifferenceDerivative
from .LazyTensors import Tensor, TensorOp, TensorPlus, TensorDot, TensorMul, TensorPow, LazyOperatorTensor