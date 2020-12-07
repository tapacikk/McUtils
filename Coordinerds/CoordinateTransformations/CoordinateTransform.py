"""
The coordinate mats class defines an architecture to mats coordinates
"""
import numpy as np
from collections import namedtuple

######################################################################################################
##
##                                   CoordinateTransform Class
##
######################################################################################################

from .TransformationFunction import TransformationFunction
from .AffineTransform import AffineTransform
from .RotationTransform import RotationTransform
from .ScalingTransform import ScalingTransform
from .TranslationTransform import TranslationTransform

__all__ = [
    "CoordinateTransform"
]

class CoordinateTransform:
    """
    The CoordinateTransform class provides a simple, general way to represent a
    compound coordinate transformation.
    In general, it's basically just a wrapper chaining together a number of TransformationFunctions.
    """

    def __init__(self, *transforms):
        if len(transforms) == 0:
            transforms = (ScalingTransform([1, 1, 1]),)
        self._transform_list = tuple(self.parse_transform(tf) for tf in transforms)
        self._trans = None
        self.condense_transforms()

    @property
    def transformation_function(self):
        """

        :return:
        :rtype: TransformationFunction
        """
        return self._trans
    @property
    def transforms(self):
        return self._transform_list

    def apply(self, coords):
        tfunc = self.transformation_function #type: TransformationFunction
        return tfunc.operate(coords)
    def __call__(self, coords):
        if isinstance(coords, (TransformationFunction, CoordinateTransform)):
            return type(self)(coords, self)
        else:
            return self.apply(coords)

    def condense_transforms(self):
        self._trans = self._transform_list[-1]
        for t in self._transform_list[:-1]:
            self._trans = self._trans.merge(t)

    @property
    def inverse(self):
        return type(self)(self._trans.inverse)

    @staticmethod
    def parse_transform(tf):
        """
        Provides a way to "tag" a transformation
        :param tf:
        :type tf:
        :return:
        :rtype:
        """

        if isinstance(tf, TransformationFunction):
            return tf
        elif isinstance(tf, CoordinateTransform):
            return tf.transformation_function
        elif isinstance(tf, str):
            # we can define some convenient transformation syntax, maybe
            raise NotImplementedError("String syntax for transformations still isn't implemented. Sorry about that.")
        else:
            tf = np.asarray(tf)
            if tf.ndim == 1:
                tf = TranslationTransform(tf)
            elif tf.shape[0] == 3 and tf.shape[1] == 1:
                tf = ScalingTransform(tf.flatten())
            else:
                tf = AffineTransform(tf)
            return tf

