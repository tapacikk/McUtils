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
            return regex.parser(txt)

        # one lingering question is how to handle `Repeating` patterns.
        # Probably want to extract the child block and feed that into parse_all...
        # might need to figure out some alternate manner for handling dtypes and block_handlers
        # in this case since I think the delegation to the parse_all might fuck stuff up...
        if isinstance(regex, RegexPattern) and regex.is_repeating:
            import copy

            iterable_regex = copy.copy(regex)
            iterable_regex.pat = lambda p:p
            parser = type(self)(iterable_regex)

            return parser.parse_all(txt)

        if block_handlers is None and isinstance(regex, RegexPattern):
            block_handlers = self.get_regex_block_handlers(regex)
        if dtypes is None and isinstance(regex, RegexPattern):
            dtypes = self.get_regex_dtypes(regex)

        res = self._set_up_result_arrays(dtypes)

        # if we're given a non-string regex we compile it
        if isinstance(regex, RegexPattern):
            regex = regex.compiled

        match = re.search(regex, txt)
        self._handle_parse_match(match, res, block_handlers)

        return res

    # we need to add some stuff to generalize out the parser, add some useful general methods
    # like the handling of blocks of unknown size / unknown numbers of atoms / efficient casting from string
    # to NumPy types
    def parse_all(self, txt,
                  regex = None,
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

        match_iter = re.finditer(regex, txt)
        if num_results is not None:
            import itertools as it
            matches = list(it.islice(match_iter, None, num_results))
        else:
            matches = list(match_iter)

        if len(matches)==0:
            raise StringParserException("Pattern {} not found".format(regex))

        self._handle_parse_match(matches, res, block_handlers, append = True)
        #  we don't need to append since we've pushed the filling iteration into the StructuredArray

        return res

    def parse_iter(self, txt,
                  regex = None,
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

        # if we're given a non-string regex we compile it
        if isinstance(regex, RegexPattern):
            regex = regex.compiled

        match_iter = re.finditer(regex, txt)
        if num_results is not None:
            import itertools as it
            match_iter = it.islice(match_iter, None, num_results)

        for match in match_iter:
            res = self._set_up_result_arrays(dtypes)
            self._handle_parse_match(match, res, block_handlers)
            yield res

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


    def _split_match_groups(self, groups, res):
        if len(groups[0]) == len(res):
            # just need to transpose the groups, basically,
            gg = groups.T
        else:
            blocks = [ r.block_size for r in res ]
            gg = [ None ] * len(blocks)
            sliced = 0
            for i, b in enumerate(blocks):
                if b == 1:
                    gg[i] = groups[:, sliced]
                else:
                    gg[i] = groups[:, sliced:sliced+b]
                sliced += b
        return gg # groups should now be chunked into groups if it wasn't before

    def _handle_parse_match(self, match, res, block_handlers, append = False,
                            default_value = ""
                            ):
        """Figures out how to handle the matched data from the parser
        In general the way we _want_ to do this is to basically do the most minimal amount of processing we can
        and then use the efficient methods inside StructuredTypeArray to cast this to properly typed arrays

        One issue is that we can end up with stuff that looks like:
            [
                [ group_a match_1, group_a match_2, group_b match_1, ... ]
                ...
            ]
        where we need to first chop the data into group_a, group_b, ...

        The type-calculus in RegexPattern and StructuredType can give us a "block size" for each subarray inside the
        StructuredTypeArray which indicates how many elements it expects to see from the group

        We use that to chop up the array as needed in these multi-match cases before trying to feed it into the
        StructuredTypeArray

        Complicating that, though, is that we sometimes need to do an extra layer of parsing on our matches

        Imagine that we found every instance of Repeating(Capturing(Number), suffix=Optional(Whitespace))

        This could give us something like
            [
                '-1232.123-234.345 3434.3434 10000000000',
                '5',
                '6'
            ]

        It would be hard to say what we should do with this, so we'll assume (possibly incorrectly) that we won't get a
        ragged array. If we need to introduce raggedness we'll do it later

        Even then, we could have
            [
                '-1232.123-234.345 3434.3434 10000000000',
                '5         5       5         5',
                '6         6       6         6'
            ]

        Here we'd need to go through line-by-line and apply the subparser of Number to each line to split it up

        On the other hand, what if we matched Duplicated(Capturing(Number), 4, riffle=Optional(Whitespace))?

        Then we'd have
            [
                ['-1232.123', '-234.345', '3434.3434', '10000000000'],
                ['5',         '5',        '5',         '5'],
                ['6',         '6',        '6',         '6']
            ]

        And here we wouldn't want to apply the subparser.

        The only way we could determine the difference is by comparing the shape of the StructuredTypeArray to the
        matched data. This means our block_handlers need to take the passed res array to figure their shit out too.


        :param match:
        :type match: re.Match | iterable[re.Match]
        :param res:
        :type res:
        :param block_handlers:
        :type block_handlers:
        :return:
        :rtype:
        """

        if hasattr(match, 'groupdict'):
            single = True
            first_match = match # indicates we were fed a single match
        else:
            single = False
            if len(match) == 0:
                raise StringParserException("Can't handle series of matches if there are no matches")
            first_match = match[0] # indicates we were fed a series of matches
            # indicates we were fed a single group

        gd = first_match.groupdict(default=default_value) if isinstance(res, OrderedDict) else []
        if len(gd) == 0:
            # we have no _named_ groups so we just pull everything and run with it
            groups = match.groups(default=default_value) if single else np.array([ m.groups(default=default_value) for m in match ], dtype=str)
            if isinstance(res, StructuredTypeArray):
                if not single and not res.is_simple:
                    # means we need to split up the groups
                    groups = self._split_match_groups(groups, res.array)

                if block_handlers is not None:
                    # means we implicitly have groups inside res... so I guess we need to zip in res.array too?
                    for b,a,g in zip(block_handlers, res.array, groups):
                        if b is not None:
                            g = b(g, array = a)
                        if single and append:
                            a.append(g)
                        elif append:
                            a.extend(g, single = single)
                        else:
                            a.fill(g)
                else:
                    if single and append:
                        res.append(groups)
                    elif append:
                        res.extend(groups, single = single)
                    else:
                        res.fill(groups)
            else:
                if isinstance(res, OrderedDict): # should I throw an error here...?
                    res_iter = res.items()
                else:
                    res_iter = res
                # we thread through the results and groups
                # if we aren't in the single match regime we need to iterate through results individually since there's no
                # way to determine what the correct shape is I think...? Or maybe we can use the result shapes to determine this?
                # let's try the latter
                if not single:
                    groups = self._split_match_groups(groups, res)

                if block_handlers is not None and not isinstance(block_handlers, (dict, OrderedDict)):
                    for s, a, h in zip(groups, res_iter, block_handlers):
                        if h is not None:
                            s = h(s, array = a)
                        if single and append:
                            a.append(s)
                        elif append:
                            a.extend(s, single = single)
                        else:
                            a.fill(s)
                else: # otherwise we just ignore the block_handlers I guess ???
                    for s, a in zip(groups, res_iter):
                        if single and append:
                            a.append(s)
                        elif append:
                            a.extend(s, single = single)
                        else:
                            a.fill(s)
        else:
            if block_handlers is None:
                block_handlers = {}
            groups = gd if single else [ m.groupdict(default=default_value) for m in match ]
            # now we need to remerge the groupdicts by name if not single
            if not single:
                groups = OrderedDict(
                    (
                        k,
                        np.array([ g[k] for g in groups ], dtype=str)
                     ) for k in groups[0]
                )

            for k, v in groups.items():

                r = res[k]
                if k in block_handlers:
                    handler = block_handlers[k]
                    if handler is not None:
                        v = handler(v, array = r)
                        # we need to push the batch processing into the handler then...?
                        # how can the handler know what to do with like arrays of stuff though?

                else:
                    # unclear what we do in this case? By default we'll just not bother to handle it
                    handler = None

                if append:
                    if handler is not None and isinstance(v, (StructuredTypeArray, np.ndarray)):
                        r.extend(v) # we'll assume the handler was smart?
                    elif (not single) or isinstance(v, np.ndarray):
                        r.extend(v, single = single)
                    else:
                        r.append(v)
                else:
                    # print(r, v)
                    r.fill(v)

    @classmethod
    def _get_regex_handler(cls, r):
        # if we have a simple dtype we just parse out the whole thing, no need for parse_all
        # otherwise if we have non-None repetitions we need parse_all
        if r.dtype.is_simple and r.dtype.shape is None:
            handler = None
        else:
            # if not r.dtype.is_simple:

            # we should only ever end up here on Named and Capturing objects
            # these will cause infinite recursion if we don't strip their Capturing status somehow
            kids = r.children
            if len(kids) == 1:
                r = kids[0]
            else:
                # we gotta find a way to mess with r so parse works right...?
                import copy
                r = copy.copy(r)
                r.pat = lambda p, *a, **k:p
                r.capturing = False
                r.invalidate_cache()

            def handler(t, r=r, cls = cls, array = None):

                if array is not None and array.can_cast(t):
                    rr = t
                elif isinstance(t, str):
                    s = cls(r)
                    rr = s.parse(t)
                else:
                    # we assume we got a list of strings...?
                    rr = None
                    s = cls(r)
                    for l in t:
                        if rr is None:
                            rr = s.parse(l)
                        else:
                            r2 = s.parse(l)
                            rr.extend(r2)
                    # somehow this is working but I don't _really_ know how...?

                return rr

        return handler

    @classmethod
    def get_regex_block_handlers(cls, regex):
        """Uses the uncompiled RegexPattern to determine what blocks exist and what handlers they should use

        :param regex:
        :type regex: RegexPattern
        :return:
        :rtype:
        """
        # I think it might make sense to allow a given regex to declare its _own_ parser that specifies how it
        # should be used in parse and parse_all. This can remain the default implementation, but that would allow
        # one to push the complexity of figuring out how to handle an object onto the object itself...

        # since we handle repeating patterns by delegating to parse_all by default, these handlers will need to be the
        # handlers for the child parse_all call instead of the basic parser from the first matched child, I think...
        # ah actually more generally we need to make sure that for each captured block (i.e. what powers parser)
        # we have a handler
        # the basic assumption has generally been that all children will be relevant, but as things
        # increase in complexity this assumption breaks down
        # therefore the regex needs to remember itself which of its children are capturing and which aren't

        handlers = regex.named_groups
        if handlers is not None:
            for k,r in tuple(handlers.items()): # we extract the full list first so we can go back and edit the handlers map
                handlers[k] = cls._get_regex_handler(r)
        else:
            # handlers = None
            # if we have no named groups to work with we'll create a block handler for the capturing groups?
            # print(repr(regex), regex.children)
            handlers = regex.capturing_groups
            if handlers is not None:
                if isinstance(handlers, RegexPattern):
                    handlers = [ handlers ]
                handlers = [ cls._get_regex_handler(r) for r in handlers if r.dtype is not DisappearingType ]

        return handlers
    @classmethod
    def get_regex_dtypes(cls, regex):
        """Uses the uncompiled RegexPattern to determine which StructuredTypes to return

        :param regex:
        :type regex: RegexPattern
        :return:
        :rtype:
        """

        dtypes = regex.named_groups
        if dtypes is not None:
            for k, r in tuple(dtypes.items()):
                dtypes[k] = r.dtype
        else:## there is some question as to whether I might actually want to make it so the _children_ get parsed out...
            dtypes = regex.capturing_groups
            if dtypes is not None:
                if isinstance(dtypes, RegexPattern):
                    dtypes = [dtypes]
                dtypes = tuple(d.dtype for d in dtypes)
            else:
                dtypes = regex.dtype

        return dtypes

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