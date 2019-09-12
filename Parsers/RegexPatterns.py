"""
Simple utilities that support constructing Regex patterns
"""

import re
from collections import OrderedDict
from .StructuredType import StructuredType, DisappearingType

__all__ = [
    "RegexPattern",
    "Capturing",
    "NonCapturing",
    "Optional",
    "Alternatives",
    "Longest",
    "Shortest",
    "Repeating",
    "Duplicated",
    "PatternClass",
    "Parenthesized",
    "Named",
    "Any",
    "Sign",
    "Number",
    "Integer",
    "PositiveInteger",
    "ASCIILetter",
    "AtomName",
    "WhitespaceCharacter",
    "Whitespace",
    "Word",
    "WordCharacter",
    "CartesianPoint",
    "IntXYZLine",
    "XYZLine",
    "Empty",
    "Newline",
    "ZMatPattern"
]

######################################################################################################################
#
#                                            RegexPattern
#
class RegexPattern:
    """
    Represents a combinator structure for building more complex regexes

    It might be worth working with this combinator structure in a _lazy_ fashion so that we can drill down
    into the expression structure... that way we can define a sort-of Regex calculus that we can use to build up higher
    order regexes but still be able to recursively inspect subparts?
    """
    def __init__(self,
                 pat,
                 name = None,
                 children = None,
                 parents = None,
                 dtype = None,
                 repetitions = None,
                 key = None,
                 joiner = "",
                 wrapper_function = None,
                 suffix = None, # often we need a start or end RegexPattern for the match, but we don't actually care the data in it
                 prefix = None,
                 parser = None # a parser function we can use instead of the default StringParser parser
                 ):
        """
        :param pat:
        :type pat: str | callable
        :param name:
        :type name: str
        :param dtype:
        :type dtype:
        :param repeated:
        :type repeated:
        :param key:
        :type key:
        :param joiner:
        :type joiner:
        :param children:
        :type children:
        :param parents:
        :type parents:
        :param wrapper_function:
        :type wrapper_function:
        """

        if isinstance(pat, (tuple, list)) and children is None:
            children = pat
            pat = lambda p:p # basically represents a compound pattern...
        self._pat = pat
        self._cached = None
        self._comp = None

        self.name = name # not used for anything but can make the code clearer

        # will some day be useful in automatic type determination in a StringParser
        self._dtype = dtype
        self.repetitions = repetitions

        self.key = key # to be used when matching to find subcomponents

        # we'll build a regex tree structure
        # that we can use to walk the regex string structure
        if children is None:
            children = []
        elif not isinstance(children, list):
            if isinstance(children, str):
                children = [ children ]
            else:
                children = list(children)

        children = [ c if isinstance(c, RegexPattern) else NonCapturing(c) for c in children ]
        for c in children:
            c.add_parent(self)
        self._children = children
        self._child_map = None
        if parents is None:
            parents = []
        elif not isinstance(parents, list):
            parents = list(parents)
        self._parents = parents # we need this so we can propagate recompile calls all the way up
        self._joiner = joiner
        self._wrapper_function = wrapper_function
        self._combine_args = ()
        self._combine_kwargs = {}

        self._prefix = prefix
        self._suffix = suffix
        self.parser = parser

    @property
    def pat(self):
        return self._pat
    @pat.setter
    def pat(self, pat):
        self._pat = pat
        self.invalidate_cache()
    @property
    def children(self):
        """

        :return:
        :rtype: tuple[RegexPattern]
        """
        return tuple(self._children)
    @property
    def child_count(self):
        """

        :return:
        :rtype: int
        """
        return len(self._children)
    @property
    def child_map(self):
        """Returns the map to subregexes for named regex components

        :return:
        :rtype: Dict[str, RegexPattern]
        """
        # we'll make a new OD every time to decrease the risk of mutability breakages
        # if self._child_map is None:
        #     self._child_map = OrderedDict( (r.key, r) for r in self._children if r.key is not None )
        return OrderedDict( (r.key, r) for r in self._children if r.key is not None )
    @property
    def parents(self):
        """

        :return:
        :rtype: tuple[RegexPattern]
        """
        return tuple(self._parents)
    @property
    def joiner(self):
        """

        :return:
        :rtype: str
        """
        return self._joiner
    @joiner.setter
    def joiner(self, j):
        self._joiner = j
        self._cached = None
    @property
    def suffix(self):
        """

        :return:
        :rtype: str | RegexPattern
        """
        return self._suffix
    @suffix.setter
    def suffix(self, e):
        self._suffix = e
        self._cached = None
    @property
    def prefix(self):
        """

        :return:
        :rtype: str | RegexPattern
        """
        return self._prefix
    @prefix.setter
    def prefix(self, s):
        self._prefix = s
        self._cached = None
    @property
    def dtype(self):
        """Returns the StructuredType for the matched object

        :return:
        :rtype: None | StructuredType
        """
        dt = self._dtype
        if dt is None and self.child_count > 0:
            subdts = [c.dtype for c in self._children]
            if len(subdts) == 1:
                dt = subdts[0] # singular type
                if not isinstance(dt, StructuredType):
                    dt = StructuredType(dt)
            else:
                import functools as ft
                from operator import add

                if not isinstance(subdts[0], StructuredType):
                    subdts[0] = StructuredType(subdts[0])
                dt = ft.reduce(add, subdts) # build the compound type
            if self.repetitions is not None:
                reps = self.repetitions
                if isinstance(reps, int):
                    reps = (reps, None)
                dt = dt.repeat(*reps)
        elif dt is not None and (not isinstance(dt, StructuredType)):
            dt = StructuredType(dt)

        return dt

    # we want a Regex component to be reusable in lots of different Regexes without having to copy all of them
    # so we'll make it so that we can only walk _down_ a regex tree, and not worry about having to go back up
    # @property
    # def siblings(self):
    #     return

    def combine(self, other, *args, **kwargs):
        """Combines self and other

            :param other:
            :type other: RegexPattern | str
            :return:
            :rtype: str | callable
            """
        op = other.pat if isinstance(other, RegexPattern) else other
        if isinstance(self.pat, str) and isinstance(op, str):
            return self.pat + op
        elif isinstance(self.pat, str):
            return lambda p, wrap = op, prefix = self.pat, a = args, kw = kwargs :prefix + wrap(p, *a, **kw)
        elif isinstance(op, str):
            return self.pat(op, *args, **kwargs)
        else:
            return lambda p, wrap = op, prewrap = self.pat, a = args, kw = kwargs :prewrap(wrap(p, *a, **kw))

    def wrap(self, *args, **kwargs):
        self._combine_args = args
        self._combine_kwargs = kwargs
        if self._wrapper_function is not None:
            self._wrapper_function(self, *args, **kwargs)

    def build(self,
              joiner = None,
              prefix = None,
              suffix = None,
              recompile = False
              ):
        if recompile or self._cached is None:
            kids = [ c.build(recompile = recompile) for c in self._children ]
            if joiner is None:
                joiner = self.joiner
            if prefix is None:
                prefix = self.prefix
            if suffix is None:
                suffix = self.suffix
            joiner = joiner.build() if isinstance(joiner, RegexPattern) else joiner if joiner is not None else ""
            prefix = prefix.build() if isinstance(prefix, RegexPattern) else prefix if prefix is not None else ""
            suffix = suffix.build() if isinstance(suffix, RegexPattern) else suffix if suffix is not None else ""
            self._cached = self.combine(
                #unclear whether I should be putting the prefix/suffix inside or outside this ._.
                prefix + joiner.join(kids) + suffix,
                *self._combine_args,
                **self._combine_kwargs
            )
        return self._cached
    @property
    def compiled(self):
        if self._comp is None:
            self._comp = re.compile(self.build())
        return self._comp

    def add_parent(self, parent):
        self._parents.append(parent)
    def remove_parent(self, parent):
        self._parents.remove(parent)

    def add_child(self, child):
        self._children.append(child)
        self._child_map = None
        self.invalidate_cache()
    def add_children(self, children):
        self._children.extend(children)
        self._child_map = None
        self.invalidate_cache()
    def remove_child(self, child):
        self._children.remove(child)
        self._child_map = None
        self.invalidate_cache()
    def insert_child(self, index, child):
        self._children.insert(index, child)
        self._child_map = None
        self.invalidate_cache()

    def invalidate_cache(self):
        # propagating this up the tree is the _only_ time I need _parents
        # which is a potential source of memory leaks... should I kill off the upwards propagation?
        self._cached = None
        self._comp = None
        for p in self._parents:
            p.invalidate_cache()

    # turns out python is clever enough to automatically handle cases like this?
    # def __del__(self):
    #     # we need this for cleanup to avoid memory leaks when people call
    #     # del ... on the Regex -- it won't work perfectly, but it'll be good enough
    #     for c in self._children:
    #         c.remove_parent(self)
    def __copy__(self):
        from copy import copy
        cls = type(self)
        new = cls.__new__(cls) # avoid init call
        new.__dict__.update(self.__dict__)
        new._children = copy(new._children) # copy the children list
        new._parents = [] # since this is a copy it has no parents
        new._cached = None # just safer...
        return new

    def __add__(self, other):
        """Combines self and other

        :param other:
        :type other: RegexPattern
        :return:
        :rtype:
        """
        if not isinstance(other, RegexPattern):
            other = RegexPattern(other)
        return type(self)(
            lambda p: p,
            children = [self, other]
        )
    def __radd__(self, other):
        """Combines self and other

        :param other:
        :type other: RegexPattern
        :return:
        :rtype:
        """
        if not isinstance(other, RegexPattern):
            other = RegexPattern(other)
        return type(self)(
            lambda p: p,
            children = [other, self]
        )
    def __call__(self, other,
                 *args,
                 # we'll treat this as an alternate constructor block
                 name = None,
                 dtype = None,
                 repetitions = None,
                 key = None,
                 joiner = None,
                 wrap_function = None,
                 suffix = None,
                 prefix = None,
                 multiline = None,
                 parser = None,
                 **kwargs
                 ):
        """Wraps self around other

        :param other:
        :type other: RegexPattern
        :return:
        :rtype:
        """
        from copy import copy
        new = copy(self)

        if name is not None:
            new.name = name
        if key is not None:
            new.key = key
        if joiner is not None:
            new._joiner = joiner
        if dtype is not None:
            new._dtype = dtype
        if repetitions is not None:
            new.repetitions = repetitions
        if wrap_function is not None:
            new._wrapper_function = wrap_function
        if suffix is not None:
            new._suffix = suffix
        if prefix is not None:
            new._prefix = prefix
        if parser is not None:
            new.parser = parser

        if isinstance(other, (tuple, list)):
            other = [ RegexPattern(o) if not isinstance(o, RegexPattern) else o for o in other ]
            new.add_children(other)
        else:
            if not isinstance(other, RegexPattern):
                other = RegexPattern(other)
            new.add_child(other)

        new.wrap(*args, **kwargs)

        return new

    def __repr__(self):
        return "{}('{}', children : <{}>, pattern : {})".format(
            type(self).__name__,
            self.name,
            len(self._children),
            self._pat
        )
    def __str__(self):
        return self.build()

    def __getitem__(self, item):
        return self.child_map[item]


    ### Supporting the re mechanisms directly for convenience
    def match(self, txt):
        return re.match(self.compiled, txt)
    def search(self, txt):
        return re.search(self.compiled, txt)
    def findall(self, txt):
        return re.findall(self.compiled, txt)
    def finditer(self, txt):
        return re.finditer(self.compiled, txt)

######################################################################################################################
#
#                                       Concrete RegexPatterns to combine and use
#
#
def is_grouped(p):
    return (p.startswith('(') and p.endswith(")")) or ( p.startswith("\\") and len(p) == 2)
def group(p):
    return r"("+p+r")"
def non_capturing(p):
    return r"(?:"+p+r")"
def optional(p):
    if not is_grouped(p):
        return non_capturing(p) + "?"
    else:
        return p + "?"
def shortest(p):
    if not p.endswith("*") | p.endswith("+"):
        if not is_grouped(p):
            return non_capturing(p) + "*?"
        else:
            return p + "*?"
    else:
        return p + "?"
def repeating(p, min = 1, max = None):
    if not is_grouped(p):
        p = non_capturing(p)
    if max is None and min is None:
        return p+"*"
    elif min == 1:
        return p+"+"
    elif max is None:
        return p +"{" + str(min) + ",}"
    elif min == max:
        return p +"{" + str(min) + "}"
    else:
        return p +"{" + str(min) + "," + str(max) + "}"
def duplicated(p, num, riffle=""):
    if isinstance(riffle, RegexPattern):
        riffle = str(riffle)
    return riffle.join([p]*num)
def named(p, n):
    return "(?P<"+n+">"+p+")"
def alternatives(p):
    return "|".join(p)

# wrapper patterns
grp_p = group # capturing group
Capturing = RegexPattern(grp_p, "Capturing")
non_cap_p = non_capturing # non-capturing group
NonCapturing = RegexPattern(non_cap_p, "NonCapturing", dtype=DisappearingType)

op_p = optional # optional group
opnb_p = lambda p: r"(?:"+p+r")?" # optional non-binding group
Optional = RegexPattern(optional,
                        "Optional"
                        )
Alternatives = RegexPattern(lambda p:p, joiner="|")

lm_p = repeating
Longest = RegexPattern(lm_p, "Longest")

sm_p = shortest
Shortest = RegexPattern(sm_p, "Shortest")

def wrap_repeats(self, min = None, max = None):
    self.repetitions = (min, max)
Repeating = RegexPattern(repeating,
                         "Repeating",
                         wrapper_function=wrap_repeats
                         )
def wrap_name(self, n):
    self.key = n
    if self.name is None:
        self.name = n
Named = RegexPattern(
    named,
    "Named",
    wrapper_function = lambda self, n: setattr(self, "key", n)
)
def wrap_duplicate_type(self, n, riffle = ""):
    dt = self._dtype
    if isinstance(dt, tuple):
        dt, shape = dt
    else:
        shape = None
    if shape is None:
        shape = (n,)
    else:
        shape = (n,) + shape
    self._dtype = (dt, shape)
Duplicated = RegexPattern(duplicated,
                          "Duplicated",
                          wrapper_function = wrap_duplicate_type
                          )

pc_p = lambda p: r"["+p+r"]" # pattern class
PatternClass = RegexPattern(pc_p, "PatternClass")

parened_p = lambda p: r"\("+p+"\)"
Parenthesized = RegexPattern(pc_p, "Parenthesized")

# raw declarative patters
any_p = "."
Any = RegexPattern(any_p, "Any")

sign_p = r"[\+\-]"
Sign = RegexPattern(sign_p, "Sign")

paren_p = r"\("+".*?"+"\)"

num_p = opnb_p(sign_p)+r"\d*\.\d+" # real number
Number = RegexPattern(num_p, "Number", dtype=float)

int_p = opnb_p(sign_p)+r"\d+" # integer
Integer = RegexPattern(int_p, "Integer", dtype=int)

posint_p = r"\d+" # only positive integer
PositiveInteger = RegexPattern(posint_p, "PositiveInteger", dtype=int)

ascii_p = "[a-zA-Z]"
ASCIILetter = RegexPattern(ascii_p, "ASCIILetter", dtype=str)

name_p = ascii_p+"{1,2}" # atom name
AtomName = RegexPattern(name_p, "AtomName", dtype=str)

ws_char_class = r"(?!\n)\s" # probably excessive... but w/e I'm not winning awards for speed here
WhitespaceCharacter = RegexPattern(ws_char_class, "WhitespaceCharacter", dtype=str)

ws_p = non_capturing(ws_char_class)+"*" # whitespace
wsr_p = non_capturing(ws_char_class)+"+" # real whitespace
Whitespace = RegexPattern(ws_p, "Whitespace", dtype=str)

WordCharacter = RegexPattern("\w", "WordCharacter", dtype=str)
Word = RegexPattern("\w+", "Word", dtype=str)

ascii_punc_char_class = r"\.,<>?/'\";:{}\[\]\+=\(\)\*&\^%$#@!~`"
ASCIIPunctuation = RegexPattern(ascii_punc_char_class, "ASCIIPunctuation", dtype=str)

cart_p = ws_p.join([ grp_p(num_p) ]*3) # cartesian coordinate
CartesianPoint = RegexPattern(cart_p, "CartesianPoint", dtype=(float, (3,)))

acart_p = "("+int_p+")"+ws_p+cart_p # atom coordinate as comes from a XYZ table
IntXYZLine = RegexPattern(acart_p, "IntXYZLine", dtype=(int, (float, (3,))))

aNcart_p = "("+name_p+")"+ws_p+cart_p # atom coordinate as comes from a XYZ table
XYZLine = RegexPattern(aNcart_p, "XYZLine", dtype=(str, (float, (3,))))

Empty = RegexPattern("", "Empty")

Newline = RegexPattern("\n", "Newline", dtype=str)

ZMatPattern = Capturing(AtomName)
for i in range(3):
    ZMatPattern += Optional(NonCapturing(
        Repeating(Whitespace, 1) + Capturing(PositiveInteger) + # ref int
        Repeating(Whitespace, 1) + Capturing(Number) # ref value
    ))
ZMatPattern.name = "ZMatPattern"
