
import numpy as np

__all__ = [
    "StructuredType",
    "StructuredTypeArray",
    "DisappearingType"
]

class StructuredType:
    """
    Represents a structured type with a defined calculus to simplify the construction of combined types when writing
    parsers that take multi-typed data
    """
    def __init__(self,
                 base_type,
                 shape = None,
                 is_alternative = False,
                 is_optional = False,
                 default_value = None
                 ):
        if shape is None and isinstance(base_type, tuple) and len(base_type) == 2 and (
                isinstance(base_type[1], int) or isinstance(base_type[1], tuple)
        ): # to make it possible to initialize the type like (str, 3) or (int, (3,))
            base_type, shape = base_type
        # if isinstance(base_type, (list, tuple)):
        #     base_type = tuple(b for b in base_type if not (b is DisappearingType or isinstance(b, DisappearingTypeClass)))
        #     base_type, shape = self._condense_types(base_type, shape)
        self.dtype = base_type
        self.shape = shape

        # there's some question as to how we should declare alternative types?
        # currently we have a flag in the type but we could also make 3 subtypes of StructuredType:
        #   AlternativeTypes(...), PrimitiveType(...), CompoundType(...) ?

        self.is_alternative = is_alternative
        self.is_optional = is_optional
        self.default = default_value # for optional types

    @property
    def is_simple(self):
        return (
                self.dtype is None or isinstance(self.dtype, type) and not self.is_optional and not self.is_alternative
        )

    def __add__(self, other):
        if other is DisappearingType or isinstance(other, type(DisappearingType)):
            return self
        elif self is DisappearingType or isinstance(self, type(DisappearingType)):
            return other
        else:
            if not isinstance(other, StructuredType):
                other = StructuredType(other)

            # we define a very simple calculus on the structured types
            #   if both are the same we simply futz with the shapes if possible
            #   if they're different we construct a compound type from the two structured types
            # _unless_ we have an alternative type in which case...?

            if self.shape is None or other.shape is None:
                shape_mismatches = None
            else:
                shape_mismatches = np.array([ (x is None) or (y is None) or int(x != y) for x,y in zip(self.shape, other.shape)])
            if (
                    self.is_simple and
                    self.dtype == other.dtype and (
                            shape_mismatches is None or (
                            len(self.shape) == len(other.shape) and np.sum(shape_mismatches) <= 1
                            )
                    )
                ):

                if self.shape is None and other.shape is None:
                    new_shape = (2,)
                elif self.shape is None:
                    new_shape = other.shape[:-1] + (other.shape[-1]+1, ) # this isn't quite right but will work for now
                elif other.shape is None:
                    new_shape = self.shape[:-1] + (self.shape[-1]+1, ) # this isn't quite right but will work for now
                else:
                    mismatch_pos = np.argwhere(shape_mismatches == 1)
                    if len(mismatch_pos) == 0:
                        mismatch_pos = -1
                    else:
                        mismatch_pos = mismatch_pos[-1]
                    new_shape = list(self.shape)
                    me = self.shape[mismatch_pos]
                    yu = other.shape[mismatch_pos]
                    if me is None or yu is None:
                        new_shape[mismatch_pos] = None
                    else:
                        new_shape[mismatch_pos] = me + yu

                import copy
                new = copy.copy(self)
                new.shape = tuple(new_shape)
                return new
            else:
                return type(self)((self, other))

    def repeat(self, n = None, m = None):
        import copy
        new = copy.copy(self)
        if new.shape is None:
            new.shape = (None,) if (n is None or m is None or n !=m ) else (n,)
        elif n is not None and m is not None and n == m:
            new.shape = (n,) + new.shape
        else:
            new.shape = (None,) + new.shape # we won't handle things like between 10 and 20 elements for now
        return new

    def _condense_types(self, base_types, shape):
        # what do I do about this existing shape...?

        raw_types = [ b.dtype for b in base_types ]
        shapes = [ b.shape for b in base_types ]

        if shape is None and (
                all(r is raw_types[0] for r in raw_types) and
                (
                        all( s is None for s in shapes ) or
                        all( isinstance(s, tuple) and s == shapes[0] for s in shapes )
                )
        ):
            # means we can actually condense them but I might want to be a bit smarter with how I handle the tuple shape check...

            base_types = base_types[0] # no differences in type
            if shapes[0] is None:
                shape = (len(shapes),)
            else:
                shape += (len(shapes),)

        return base_types, shape


    def __repr__(self):
        return "{}({}, shape={})".format(
            type(self).__name__,
            self.dtype,
            self.shape
        )

class DisappearingTypeClass(StructuredType):
    """
    A special type that is entirely ignored in the structured type algebra
    """
    def __init__(self):
        self.is_disappearing = True
        super().__init__(None)
DisappearingType = DisappearingTypeClass() # redefinition but it should be a singleton anyway

class StructuredTypeArray:
    """
    Represents an array of objects defined by the StructuredType spec provided
    mostly useful as it dispatches to NumPy where things are simple enough to do so
    """

    # at some point this should make use of the more complex structured dtypes that NumPy provides...
    def __init__(self, stype, num_elements = 50):
        """
        :param stype:
        :type stype: StructuredType
        :param num_elements: number of default elements in dynamically sized arrays
        :type num_elements: int
        """
        self._is_simple = stype.is_simple
        self._extend_along = None
        self.stype = stype
        self._array = self.empty_array(num_elements)
        self._filled_to = 0

    @property
    def is_simple(self):
        """Just returns wheter the core datatype is simple

        :return:
        :rtype:
        """
        return self._is_simple
    @property
    def extension_axis(self):
        """Determines which axis to extend when adding more memory to the array
        :return:
        :rtype:
        """
        if self._extend_along is None:
            if self.stype.shape is None:
                self._extend_along = 0
            else:
                shape_nones = np.array([x is None for x in self.stype.shape], dtype=bool)
                if len(shape_nones) == 0:
                    self._extend_along = 0
                else:
                    self._extend_along = np.arange(len(shape_nones))
                    hmm = self._extend_along[shape_nones]
                    if len(hmm) == 0:
                        hmm = self._extend_along
                    self._extend_along = hmm[0]
        return self._extend_along
    @extension_axis.setter
    def extension_axis(self, ax):
        self._extend_along = ax
    @property
    def shape(self):
        if self.is_simple:
            self._shape = list(self._array.shape)
            self._shape[self.extension_axis] = self._filled_to
            self._shape = tuple(self._shape)
        else:
            self._shape = [s.shape for s in self._array]
        return self._shape
    @shape.setter
    def shape(self, s):
        self._shape = s
    @property
    def dtype(self):
        return self.stype.dtype
    @property
    def array(self):
        if self.is_simple:
            return self._array[:self._filled_to]
        else:
            return self._array

    @property
    def filled_to(self):
        if self._is_simple:
            return self._filled_to
        else:
            return [s.filled_to for s in self._array]
    def __len__(self):
        return self._filled_to

    def empty_array(self, num_elements = 50):
        """Creates empty arrays with (potentially) default elements

        :param num_elements:
        :type num_elements:
        :return:
        :rtype:
        """
        stype = self.stype
        dt = stype.dtype
        if stype.is_simple:
            shape = stype.shape
            if shape is None:
                if stype.default is None:
                    arr = np.empty((num_elements,), dtype=dt)
                else:
                    arr = np.full((num_elements,), stype.default, dtype=dt)
            else:
                if any(x is None for x in shape):
                    # means we have an array of indeterminate size in that dimension
                    shape = tuple(num_elements if x is None else x for x in shape)
                # might want to add a check to see if dt is valid fo np.array
                if stype.default is None:
                    arr = np.empty(shape, dtype=dt)
                else:
                    arr = np.full(shape, stype.default, dtype=dt)
            return arr
        else:
            # means we gotta do this recursively and the shape doesn't even matter
            dt = [StructuredType(s) if not isinstance(s, StructuredType) else s for s in dt]
            return tuple( StructuredTypeArray(s) for s in dt )

    def extend_array(self):
        array = self._array
        if isinstance(array, np.ndarray):
            ax = self.extension_axis
            empty = self.empty_array() # should effectively double array size?
            self._array = np.concatenate(
                (
                    array,
                    empty
                ),
                axis=ax
            )
        else:
            for arr in array:
                arr.extend_array()

    def __setitem__(self, key, value):
        """Recursively sets parts of an array if not simple, otherwise just delegates to NumPy

        :param key:
        :type key:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        if self.is_simple:
            # should we do some type coercion if we're fed a string?
            if self._array.shape[0] == key:
                self.extend_array()
            self._array[key] = value
            if isinstance(key, int) and key == self._filled_to:
                self._filled_to += 1
        else:
            if isinstance(key, (int, str)):
                key = [ key ] * len(self._array)
            if len(value) > len(self._array):
                # means we need to reshape our value arrays since it came from a groups call but groups just returns flat values,
                # not the actual grouping we might want
                shapes = self.shape
                # since we're assigning to a slice I guess this means we're assigning the shape of element 2?
                # so we can just split it by the number of elements we need from shape[1:] and then use np.reshape on that
                # ah but we have pre-populated a number of the np.array rows so really we need to go from shape[2:]
                old_value = value
                value = [None]*len(shapes)
                for i, s in enumerate(shapes):
                    sub_shape = s[1:]
                    s_num = np.product(sub_shape) # get the number of elements
                    if s_num == 1:
                        value[i] = old_value[0]
                        old_value = old_value[1:]
                    else:
                        v = old_value[:s_num]
                        old_value = old_value[s_num:]
                        value[i] = np.reshape(np.array(v, dtype=object), sub_shape)

            for a, k, v in zip(self._array, key, value):
                a[k] = v
    def __getitem__(self, item):
        """If simple, delegates to NumPy, otherwise tries to recursively get parts...?
        Unclear how slicing is best handled here.

        :param item:
        :type item:
        :return:
        :rtype:
        """
        if self.is_simple:
            return self._array[item]
        else:
            if isinstance(item, tuple):
                compound_index = True
                first_thingy = item[0]
            else:
                compound_index = False
                first_thingy = item
            if isinstance(first_thingy, slice) and compound_index:
                bits = self._array[first_thingy]
                return [ b[item[1:]] for b in bits ]
            elif isinstance(first_thingy, slice):
                raise NotImplemented("Not sure how I want to slice StructuredType objects yet")
            elif compound_index:
                bit = self._array[first_thingy]
                return bit[item[1:]]
            else:
                bit = self._array[first_thingy]
                return bit

    def add_axis(self, which = 0, num_elements = None):

        if self.is_simple:
            if self.stype.shape is not None:
                # if we've already got a flexible shape I think we actually dont need to add an axis...
                # otherwise we add an axis where specified, but in practice which will never not be 0
                shape = self._array.shape # just broadcast the numpy array
                if num_elements is None:
                    num_elements = 50 # hard coded for now but could be changed
                new_shape_1 = shape[:which] + (1,) + shape[which:] # we gotta broadcast to the 1 first for some corner cases
                self._array = np.broadcast_to(self._array, new_shape_1)
                new_shape_2 = shape[:which] + (num_elements,) + shape[which:] # now we can fully broadcast
                self._array = np.broadcast_to(self._array, new_shape_2).copy()

                import copy
                self.stype = copy.copy(self.stype)
                if self.stype.shape is None:
                    self.stype.shape = (None,)
                else:
                    self.stype.shape = self.stype.shape[:which] + (None,) + self.stype.shape[which:]
        else:
            # and if
            # we'll recursively add axes to the subarrays I think...
            # this might not be entirely in keeping with how a np.ndarray works but I think is acceptable
            for a in self._array:
                a.add_axis(which, num_elements)

            # raise NotImplementedError("Not sure yet how I want to broadcast compound type arrays to different shapes...")


    def append(self, val):
        """Puts val in the first empty slot in the array

        :param val:
        :type val:
        :return:
        :rtype:
        """
        self[self.filled_to] = val
        # self._filled_to+=1 # handled automatically by a small bit of cleverness in the filling code

    def fill(self, array):
        """Sets the result array to be the passed array

        :param array:
        :type array: str | np.ndarray
        :return:
        :rtype:
        """
        if isinstance(array, str):
            array = self.cast_to_array(array)

        if self._is_simple:
            self._array = array
            self._filled_to = len(self._array)
        else:
            for arr, data in array:
                arr.fill(data)

    def cast_to_array(self, txt):
        """Casts a string of things with a given data type to an array of that type and does some optional
        shape coercion

        :param txt:
        :type txt: str | iterable[str]
        :return:
        :rtype:
        """
        if self.is_simple:
            if len(txt.strip()) == 0:
                arr = np.array([], dtype=self.stype)
            else:
                import io
                arr = np.loadtxt(io.StringIO(txt), dtype=self.stype)
                shape = np.array(self.shape)
                if shape is not None:
                    arr = arr.flatten()
                    num_els = len(arr)
                    # the number of elements that the length of the parsed out array must be divisible by
                    num_not_along_axis = np.product(shape) / shape[self.extension_axis]
                    if num_els % num_not_along_axis == 0:
                        # means we can cleanly reshape it once we know the target shape
                        shape[self.extension_axis] = num_els / num_not_along_axis
                        arr = arr.reshape(shape)
                    # should we raise an error if not?

        else: #we'll assume we got some block of strings since there's no reason to put a parser here...
            arr = [a.cast_to_array(t) for a, t in zip(self._array, txt)]

        return arr

    def __repr__(self):
        return "{}(shape={}, dtype={})".format(
            type(self).__name__,
            self.shape,
            self.dtype
        )


