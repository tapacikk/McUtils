import numpy as np, re
from collections import OrderedDict
from .RegexPatterns import *
from .StructuredType import *

__all__ = [
    "StringParser",
    "StringParserException"
]

########################################################################################################################
#
#                                           StringParser
#
class StringParserException(Exception):
    ...

class StringParser:
    """
    A convenience class that makes it easy to pull blocks out of strings and whatnot
    """

    def __init__(self, regex):
        self.regex = regex

    # The real goal is to be able to declare some complex regex with named chunks like:
    #
    #   RegularExpression(
    #       Named(PositiveInteger, "Header") + Newline +
    #       Named(Repeated(Any), "Comment") + Newline
    #       Named(Repeating(Repeating(CartesianPoint, 3, 3) + Newline), "Atoms"),
    #       "XYZ"
    #   )
    #
    # and have StringParser be able to just _use_ it to get the appropriate data out of the file
    # that might be a bit down the road but it's possible and definitely worth trying to do

    def parse(self,
              txt,
              regex = None,
              block_handlers = None,
              dtypes = None
              ):
        """Finds a single match for the and applies parsers for the specified regex in txt

        :param txt:
        :type txt: str
        :param regex:
        :type regex:
        :return:
        :rtype:
        """

        # we'll enforce that the return type be a StructuredTypeArray
        # the enforcement mechanism here will be basically to take the Regex dtype if provided and otherwise
        # to coerce to a StructuredType

        if regex is None:
            regex = self.regex

        if isinstance(regex, RegexPattern) and regex.parser is not None:
            return regex.parser.parse_iter(txt)

        # one lingering question is how to handle `Repeating` patterns.
        # Probably want to extract the child block and feed that into parse_iter...
        # might need to figure out some alternate manner for handling dtypes and block_handlers
        # in this case since I think the delegation to the parse_iter might fuck stuff up...
        if isinstance(regex, RegexPattern) and isinstance(regex.repetitions, tuple):
            import copy

            iterable_regex = copy.copy(regex)
            iterable_regex.pat = lambda p:p
            parser = type(self)(iterable_regex)

            return parser.parse_iter(txt)

        if block_handlers is None and isinstance(regex, RegexPattern):
            block_handlers = self.get_regex_block_handlers(regex)
        if dtypes is None and isinstance(regex, RegexPattern):
            dtypes = self.get_regex_dtypes(regex)

        res = self._set_up_result_arrays(dtypes)

        # if we're given a non-string regex we compile it
        if isinstance(regex, RegexPattern):
            regex = regex.compiled

        match = re.search(regex, txt)
        res = self._handle_parse_match(match, res, block_handlers)

        return res

    # we need to add some stuff to generalize out the parser, add some useful general methods
    # like the handling of blocks of unknown size / unknown numbers of atoms / efficient casting from string
    # to NumPy types
    def parse_iter(self, txt, regex = None,
                   num_results = None,
                   block_handlers = None,
                   dtypes = None
                   ):

        if regex is None:
            regex = self.regex

        if block_handlers is None and isinstance(regex, RegexPattern):
            block_handlers = self.get_regex_block_handlers(regex)
        if dtypes is None and isinstance(regex, RegexPattern):
            dtypes = self.get_regex_dtypes(regex)

        res = self._set_up_result_arrays(dtypes)
        if isinstance(res, StructuredTypeArray):
            res.add_axis()
        elif isinstance(res, (dict, OrderedDict)):
            for r in res.values():
                r.add_axis()
        else:
            for r in res:
                r.add_axis()

        # if we're given a non-string regex we compile it
        if isinstance(regex, RegexPattern):
            regex = regex.compiled

        block_num = -1
        for block_num, match in enumerate(re.finditer(regex, txt)):
            self._handle_parse_match(match, res, block_handlers, append = True)
            if block_num == num_results:
                break

        if block_num == -1:
            raise StringParserException("Pattern {} not found".format(regex))

        return res

    def _set_up_result_arrays(self, dtypes):
        # we'll always force our dtypes and results to be a StructuredType and StructuredTypeArray
        # as these will force the results to be conformaing
        if isinstance(dtypes, type):
            dtypes = StructuredType(dtypes)
            res = StructuredTypeArray(dtypes) # type: StructuredTypeArray
        elif isinstance(dtypes, (dict, OrderedDict)):
            for k in dtypes.keys():
                if not isinstance(dtypes[k], StructuredType):
                    dtypes[k] = StructuredType(dtypes[k])
            res = OrderedDict((k, StructuredTypeArray(d)) for k, d in dtypes.items()) # type: OrderedDict[str, StructuredTypeArray]
        elif isinstance(dtypes, StructuredType):
            res = StructuredTypeArray(dtypes)
        else:
            dtypes = [ dt if isinstance(dt, StructuredType) else StructuredType(dt) for dt in dtypes ]
            res = [ StructuredTypeArray(d) for d in dtypes ] # type: list[StructuredTypeArray]

        return res
    def _handle_parse_match(self, match, res, block_handlers, append = False):
        """Figures out how to handle a single match from the parser

        :param match:
        :type match:
        :param res:
        :type res:
        :param block_handlers:
        :type block_handlers:
        :return:
        :rtype:
        """
        gd = match.groupdict()
        if len(gd) == 0:
            groups = match.groups()
            if isinstance(res, StructuredTypeArray):
                # we expect to have a single main type
                if append:
                    res.append(groups)
                else:
                    res.fill(groups)
            else:
                if isinstance(res, OrderedDict): # should I throw an error here...?
                    it = res.items()
                else:
                    it = res
                # we thread through the results and groups
                if block_handlers is not None and not isinstance(block_handlers, (dict, OrderedDict)):
                    for s, a, h in zip(groups, it, block_handlers):
                        if h is not None:
                            s = h[s]
                        if append:
                            a.append(s)
                        else:
                            a.fill(s)
                else: # otherwise we just ignore it???
                    for s, a in zip(groups, it):
                        if append:
                            a.append(s)
                        else:
                            a.fill(s)
        else:
            if block_handlers is None:
                block_handlers = {}
            for k, v in gd.items():
                if k in block_handlers:
                    handler = block_handlers[k]

                    # a convenience syntax in case it's useful...
                    if isinstance(handler, dict):
                        handler = dict({'single_line':False, 'dtype':None, 'pattern':None}, **handler)
                    elif isinstance(handler, RegexPattern):
                        handler = {'single_line':True, 'dtype':None, 'pattern':handler}

                    if isinstance(handler, dict):
                        sl = handler['single_line']
                        dt = handler['dtype']
                        pt = handler['pattern']
                        if sl:
                            v = self.parse_iter(v, pt, dtypes = dt)
                        else:
                            v = self.parse(v, pt, dtypes = dt)
                    elif handler is not None:
                        v = handler(v)
                else:
                    # unclear what we do in this case? By default we'll just not bother to handle it
                    ...

                if isinstance(v, StructuredTypeArray): # we'll assume the handler did it all for us?
                    res[k] = v
                elif append:
                    res[k].append(v)
                else:
                    res[k].fill(v)

    @classmethod
    def get_regex_block_handlers(cls, regex):
        """Uses the uncompiled RegexPattern to determine what blocks exist and what handlers they should use

        :param regex:
        :type regex: RegexPattern
        :return:
        :rtype:
        """
        import itertools as it

        # I think it might make sense to allow a given regex to declare its _own_ parser that specifies how it
        # should be used in parse and parse_iter. This can remain the default implementation, but that would allow
        # one to push the complexity of figuring out how to handle an object onto the object itself...
            
        handlers = regex.child_map
        if len(handlers) > 0:
            for k,r in tuple(handlers.items()): # we extract the full list first so we can go back and edit the handlers map
                # if we have a simple dtype we just parse out the whole thing, no need for parse_iter
                # otherwise if we have non-None repetitions we need parse_iter

                if r.dtype.is_simple:
                    handler = None#lambda t, r=r, s=s: s.parse(t)
                else:
                    # we'll assume this is named object where the first child is the relevant one
                    # there's like a 4% chance this works well
                    r = r.children[0]
                    s = cls(r)
                    handler = lambda t, r=r, s=s: s.parse(t)
                handlers[k] = handler

            if regex.key is not None:
                handler = lambda t: t
                handlers = OrderedDict(
                    ((regex.key, handler),),
                    **handlers
                )
        else:
            handlers = None
            # handlers = regex.children
            # handlers = tuple(
            #     None if r.dtype.is_simple else (lambda t, r=r, s=cls(r): s.parse_iter(t)) for r in handlers
            # )

        return handlers
    @classmethod
    def get_regex_dtypes(cls, regex):
        """Uses the uncompiled RegexPattern to determine which StructuredTypes to return

        :param regex:
        :type regex: RegexPattern
        :return:
        :rtype:
        """

        dtypes = regex.child_map
        if len(dtypes) > 0:
            for k, r in tuple(dtypes.items()):
                dtypes[k] = r.dtype

            if regex.key is not None:
                dtypes = OrderedDict(
                    ((regex.key, regex.dtype),),
                    **dtypes
                )
        else:
            dtypes = regex.dtype

            ### there is some question as to whether I might actually want to make it so the _children_ get parsed out...
            # dtypes = regex.children
            # dtypes = tuple(d.dtype for d in dtypes)

        return dtypes

    def pull_all(self, txt):


    ### Simple methods for pulling common data types out of strings
    @classmethod
    def pull_coords(cls, txt, regex = CartesianPoint.compiled, coord_dim = 3):
        """Pulls just the Cartesian coordinates out of the string

        :param txt: some string that has Cartesians in it somewhere
        :type txt:
        :param regex: the Cartesian matching regex; can be swapped out for a different matcher
        :type regex:
        :param coord_dim: the dimension of the pulled coordinates (used to reshape)
        :type coord_dim:
        :return:
        :rtype:
        """
        coords = re.findall(regex, txt)
        base_arr = np.array(coords, dtype=np.str).astype(dtype=np.float64)
        if len(base_arr.shape) == 1:
            num_els = base_arr.shape[0]
            new_shape = (int(num_els/coord_dim), coord_dim) # for whatever weird reason this wasn't int...?
            new_arr = np.reshape(base_arr, new_shape)
        else:
            new_arr = base_arr
        return new_arr

    @classmethod
    def pull_xyz(cls, txt, num_atoms = None, regex = IntXYZLine.compiled):
        """Pulls XYX-type coordinates out of a string

        :param txt: XYZ string
        :type txt: str
        :param num_atoms: number of atoms if known
        :type num_atoms: int
        :return: atom types and coords
        :rtype:
        """

        i = - 1
        if num_atoms is None:
            num_cur = 50 # probably more than we'll actually need...

            atom_types = [None]*num_cur
            coord_array = np.zeros((num_cur, 3))
            for i, match in enumerate(re.finditer(regex, txt)):
                if i == num_cur:
                    atom_types.extend([None]*(2*num_cur))
                    coord_array = np.concatenate((coord_array, np.zeros((2*num_cur, 3))), axis=0)

                g = match.groups()
                atom_types[i] = g[:-3]
                coord_array[i] = np.array(g[-3:], dtype=np.str).astype(np.float64)

        else:
            atom_types = [None]*num_atoms
            coord_array =  np.zeros((num_atoms, 3), dtype=np.float64)
            parse_iter = re.finditer(regex, txt)
            for i in range(num_atoms):
                match = next(parse_iter)
                g = match.groups()
                atom_types[i] = g[:-3]
                coord_array[i] = np.array(g[-3:], dtype=np.str).astype(np.float64)

        if i == -1:
            raise StringParserException("Pattern {} not found".format(regex))
        atom_types = atom_types[:i+1]
        coord_array = coord_array[:i+1]

        return (atom_types, coord_array)

    @classmethod
    def pull_zmat_coords(cls, txt, regex = Number.compiled):
        '''Pulls only the numeric coordinate parts out of a Z-matrix (drops ordering or types)

        :param txt:
        :type txt:
        :param regex:
        :type regex:
        :return:
        :rtype:
        '''
        coords = re.findall(regex, txt)
        # print(txt)
        # print(coords)
        base_arr = np.array(coords, dtype=np.str).astype(dtype=np.float64)
        num_els = base_arr.shape[0]
        if num_els == 1:
            base_arr = np.concatenate((base_arr, np.zeros((2,), dtype=np.float64)))
            num_els = 3
        elif num_els == 3:
            base_arr = np.concatenate((base_arr, np.zeros((1,), dtype=np.float64)))
            base_arr = np.insert(base_arr, 1, np.zeros((2,), dtype=np.float64))
            num_els = 6
        else:
            base_arr = np.insert(base_arr, 3, np.zeros((1,), dtype=np.float64))
            base_arr = np.insert(base_arr, 1, np.zeros((2,), dtype=np.float64))
            num_els = num_els + 3

        # print(base_arr)
        coord_dim = 3
        new_shape = (int(num_els/coord_dim), coord_dim)
        new_arr = np.reshape(base_arr, new_shape)
        return new_arr

    @classmethod
    def process_zzzz(cls, i, g, atom_types, index_array, coord_array, num_header=1):
        g_num = len(g)
        # print(g, g_num, num_header)
        if g_num == num_header+0:
            atom_types[i] = g
        elif g_num == num_header+2:
            atom_types[i] = g[:-2]
            # make atom refs to insert into array
            ref = np.array(g[-2:-1], dtype=np.str).astype(np.int8)
            ref = np.concatenate((ref, np.zeros((2,), dtype=np.int8)))
            coord = np.array(g[-1:], dtype=np.str).astype(np.float64)
            coord = np.concatenate((coord, np.zeros((2,), dtype=np.float64)))
            index_array[i-1] = ref
            coord_array[i-1] = coord
        elif g_num == num_header+4:
            atom_types[i] = g[:-4]
            # make atom refs to insert into array
            ref = np.array(g[-4::2], dtype=np.str).astype(np.int8)
            ref = np.concatenate((ref, np.zeros((1,), dtype=np.int8)))
            coord = np.array(g[-3::2], dtype=np.str).astype(np.float64)
            coord = np.concatenate((coord, np.zeros((1,), dtype=np.float64)))
            index_array[i-1] = ref
            coord_array[i-1] = coord
        else:
            atom_types[i] = g[:-6]
            index_array[i-1] = np.array(g[-6::2], dtype=np.str).astype(np.int8)
            coord_array[i-1] = np.array(g[-5::2], dtype=np.str).astype(np.float64)

    @classmethod
    def pull_zmat(cls,
                  txt,
                  num_atoms = None,
                  regex = ZMatPattern.compiled,
                  num_header = 1
                  ):
        """Pulls coordinates out of a zmatrix

        :param txt:
        :type txt:
        :param num_atoms:
        :type num_atoms: int | None
        :param regex:
        :type regex:
        :return:
        :rtype:
        """

        if num_atoms is None:
            num_cur = 50 # probably more than we'll actually need...

            atom_types = [None]*num_cur
            index_array = np.zeros((num_cur-1, 3))
            coord_array = np.zeros((num_cur-1, 3))
            i = -1
            for i, match in enumerate(re.finditer(regex, txt)):
                if i == num_cur:
                    atom_types.extend([None]*(2*num_cur))
                    coord_array = np.concatenate(
                        (coord_array, np.zeros((2*num_cur, 3))),
                        axis=0
                    )
                    index_array = np.concatenate(
                        (index_array, np.zeros((2 * num_cur, 3))),
                        axis=0

                    )

                g = match.groups()
                if i == 0:
                    g = g[:-6]
                elif i == 1:
                    g = g[:-4]
                elif i == 2:
                    g = g[:-2]
                # print(g)
                cls.process_zzzz(i, g, atom_types, index_array, coord_array, num_header=num_header)
            if i == -1:
                raise StringParserException("Pattern {} not found".format(regex))
            atom_types = atom_types[:i+1]
            index_array = index_array[:i]
            coord_array = coord_array[:i]

        else:
            atom_types = [None]*num_atoms
            index_array = np.zeros((num_atoms-1, 3))
            coord_array = np.zeros((num_atoms-1, 3))
            parse_iter = re.finditer(regex, txt)
            for i in range(num_atoms):
                match = next(parse_iter)
                g = match.groups()
                cls.process_zzzz(i, g, atom_types, index_array, coord_array, num_header=num_header)

        return (atom_types, index_array, coord_array)