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
            iterable_regex.pat = lambda p, *a, **kw: p # to unbind the Repeating capture screwups
            parser = type(self)(iterable_regex)

            res = parser.parse_all(txt)

        else:
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

        # one issue here is that by default this would return a list or some structured object like that
        # sometimes we don't _want_ that list
        # if we have something like Repeating(Capturing(Number), ...)
        # we don't really want [ StructuredTypeArray(...) ] returned
        # what we'd really like to get back is just the StructuredTypeArray
        # now here's the question, is there a case where if we have [ StructuredTypeArray ]
        # that we _wouldn't_ just want the array? My sense is no since StructuredType should simplify the
        # singular type by default...?
        # Or will this be unique to Repeating patterns?

        # Eh, we'll assume this is mis-structured since we've taken a basic RegexPattern and
        # then captured a repeating portion of it or some set of repeating parts of it but
        # didn't bother to simplify to just that Capturing part of the RegexPattern...?
        if not isinstance(res, (OrderedDict, StructuredTypeArray)) and len(res) == 1:
            res = res[0]

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

    #region RegexPattern data extractors
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
                    dtypes = dtypes.dtype
                else:
                    dtypes = tuple(d.dtype for d in dtypes)

            else:
                dtypes = regex.dtype

        return dtypes
    #endregion

    #region StructuredTypeArray interface

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

    #endregion

    #region Match Handler
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

            # one difficulty about this setup is that it makes vector data into a 2D array
            # on the other hand I don't think there's anything else to be done here?
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
            else: # we have no named groups and our res is a non-singular object
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
            # we have named groups so we only use the data associated with them
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

                else:
                    # unclear what we do in this case? By default we'll just not bother to handle it
                    handler = None

                if append:
                    if handler is not None:# and isinstance(v, (StructuredTypeArray, np.ndarray)):
                        r.extend(v) # we'll assume the handler was smart?
                    elif not single:
                        r.extend(v, single = single)
                    else:
                        r.append(v)
                else:
                    r.fill(v)

    @classmethod
    def _get_regex_handler(cls, r):
        # if we have a simple dtype we just parse out the whole thing, no need for parse_all
        # otherwise if we have non-None repetitions we need parse_all

        if r.handler is not None:
            return r.handler

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
                r.pat = lambda p, *a, **k : p
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
                    # at some point this will be pushed into a parse_vector
                    # function or something where we're more efficient
                    # about how we do the parsing and data caching
                    for l in t:
                        # we gotta basically go line-by-line and add our stuff in
                        # the way we do this is by applying parse to each one of these things
                        # then calling append to add all the data in
                        if rr is None:
                            rr = s.parse(l)
                            # now that we have this object we need to basically add an axis to it
                            # the raw thing will be lower-dimensional that we want I think
                            # the way we handle that is then...? I guess hitting it all with an add_axis?
                            if isinstance(rr, StructuredTypeArray):
                                rr.add_axis()
                            elif isinstance(rr, OrderedDict):
                                for k in rr:
                                    rr[k].add_axis()
                            else:
                                for r in rr:
                                    r.add_axis()

                        else:
                            r2 = s.parse(l)
                            # we might have an OrderedDict or we might have a list or we might have a StructuredArray
                            # gotta make sure all of these cases are clean
                            if isinstance(rr, StructuredTypeArray):
                                rr.append(r2)
                            elif isinstance(rr, OrderedDict):
                                for k in rr:
                                    rr[k].append(r2[k])
                            else:
                                for r1, r2 in zip(rr, r2):
                                    r1.append(r2) # should I be using extend...?
                return rr

        return handler

    #endregion

    #region HandlerMethods
    @classmethod
    def handler_method(cls, method):
        """Turns a regular function into a handler method by adding in (and ignoring) the array argument

        :param method:
        :type method:
        :return:
        :rtype:
        """

        def handler(data, method = method, array = None):
            return method(data)

        return handler

    @classmethod
    def to_array(cls,
                 data,
                 array = None,
                 dtype = 'float',
                 shape = None
                 ):
        """A method to take a string or iterable of strings and quickly dump it to a NumPy array of the right dtype (if it can be cast as one)

        :param data:
        :type data:
        :param dtype:
        :type dtype:
        :return:
        :rtype:
        """
        import io

        if not isinstance(data, str):
            data = '\n'.join((d.strip() for d in data))
        arr = np.loadtxt(io.StringIO(data), dtype=dtype)
        if shape is not None:
            fudge_axis = [ i for i, s in enumerate(shape) if s is None]
            if len(fudge_axis) == 0:
                arr = arr.reshape(shape)
            elif len(fudge_axis) == 1:
                fudge_axis = fudge_axis[0]
                num_els = np.product(arr.shape)
                shoop = list(shape)
                shoop[fudge_axis] = 1
                num_not_on_axis = np.product(shoop)
                if num_els % num_not_on_axis == 0:
                    num_left = num_els // num_not_on_axis
                    shoop[fudge_axis] = num_left
                    arr = arr.reshape(shoop)
                else:
                    raise StringParserException("{}.{}: can't reshape array with shape '{}' to shape '{}' as number of elements is not divisible by block size".format(
                        cls.__name__,
                        'to_array',
                        arr.shape,
                        shape
                    ))
            else:
                raise StringParserException("{}.{}: can't reshape to shape '{}' with more than one indeterminate slot".format(
                    cls.__name__,
                    'to_array',
                    shape
                ))

        return arr
    @classmethod
    def array_handler(cls,
                      array = None,
                      dtype = 'float',
                      shape = None
                      ):
        """Returns a handler that uses to_array

        :param dtype:
        :type dtype:
        :param array:
        :type array:
        :param shape:
        :type shape:
        :return:
        :rtype:
        """
        def handler(data, array = array, dtype = dtype, shape = shape, cls=cls):
            return cls.to_array(data, array = array, dtype = dtype, shape = shape)
        return handler
    #endregion