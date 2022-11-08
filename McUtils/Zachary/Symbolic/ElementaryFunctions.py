
import abc, numpy as np, itertools, collections, copy
from functools import reduce

from ...Combinatorics import SymmetricGroupGenerator
from  ...Misc import Abstract


__all__ = [
    "Symbols"
    # "Summation",
    # "Product"
]
class Functionlike(metaclass=abc.ABCMeta):
    """
    A function suitable for symbolic manipulation
    with derivatives and evlauation
    """

    @abc.abstractmethod
    def eval(self, r: np.ndarray) -> 'np.ndarray':
        ...

    compile_vars = 0
    @staticmethod
    def cur_var():
        return Functionlike.compile_vars
    @staticmethod
    def inc_var():
        v = Functionlike.compile_vars
        Functionlike.compile_vars += 1
        return v
    @staticmethod
    def reset_var():
        Functionlike.compile_vars = 0
    @staticmethod
    def get_compile_var():
        return Abstract.Name("var_"+str(Functionlike.inc_var()))
    def get_compile_spec(self)->'Abstract.Expr':
        raise NotImplementedError("don't know how to compile {}".format(self))
    # def get_eval_ast(self)->'ast.Lambda':
    #     return ast.fix_missing_locations(ast.Lambda(
    #             args=ast.arguments(
    #                 posonlyargs=[],
    #                 args=[ast.arg(arg='x', annotation=None, type_comment=None)],
    #                 vararg=None,
    #                 kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
    #             ),
    #             body=self.get_compile_spec()
    #         ))
    # def lambda_wrap(self, var, expr)->'ast.Lambda':
    #     return ast.Lambda(
    #         args=ast.arguments(
    #             posonlyargs=[],
    #             args=[ast.arg(arg=var, annotation=None, type_comment=None)],
    #             vararg=None,
    #             kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
    #         ),
    #         body=expr
    #     )
    # @staticmethod
    # def ast_var(var):
    #     return ast.Name(id=var, ctx=ast.Load())

    # @staticmethod
    # def ast_call(fn, *args, **kwargs):
    #     return ast.Call(
    #         func=fn,
    #         args=[*args],
    #         keyword=[
    #             ast.keyword(arg=k, value=v)
    #             for k,v in kwargs.items()
    #         ]
    #     )
    # @classmethod
    # def ast_attr(cls, obj, attr):
    #     if isinstance(obj, str):
    #         obj = cls.ast_var(obj)
    #     return ast.Attribute(value=obj, attr=attr, ctx=ast.Load())
    # @staticmethod
    # def ast_np(fun, *args, **kwargs):
    #     return ast.Call(
    #         func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr=fun, ctx=ast.Load()),
    #         args=[*args],
    #         keyword=[
    #             ast.keyword(arg=k, value=v)
    #             for k,v in kwargs.items()
    #         ]
    #     )
    # @classmethod
    # def ast_nparray(cls, *args):
    #     return cls.ast_np('array', ast.List(elts=[*args], ctx=ast.Load()))

    def compile(self, mode='numba'):
        self.reset_var()
        expr = eval(
            compile(
                self.get_compile_spec().to_eval_expr(),
                "<expression>", mode='eval'
            ),
            {'np':np}
        )
        if mode=='numba':
            from ...Misc import njit
            expr = njit(expr)
        return expr
    @abc.abstractmethod
    def deriv(self, *which, simplify=True) -> 'Functionlike':
        ...

    def __call__(self, r):
        if isinstance(r, Functionlike):
            return self.compose(r)
        else:
            r = np.asanyarray(r)
            return self.eval(r)

    def __neg__(self):
        return -1*self
    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            other = Scalar(other)
        return Product.construct(self, other)
        # if self.idx is None or other.idx is None
        # return Product(self, other)
    def __rmul__(self, other):
        return self * other
    def __truediv__(self, other):
        return self * Power(-1).compose(other)
    def __rtruediv__(self, other):
        return other * Power(-1).compose(self)

    def __add__(self, other):
        return Summation.construct(self, other)
    def __radd__(self, other):
        return Summation.construct(self, other)
    def __sub__(self, other):
        return Summation.construct(self, -other)
    def __rsub__(self, other):
        return Summation.construct(other, -self)

    def __pow__(self, power, modulo=None):
        return Power(power).compose(self)

    def invert(self):
        raise NotImplementedError("{} is not invertible".format(self))

    def __invert__(self):
        return self.invert()

    def copy(self):
        return copy.copy(self)

    def compose(self, other):
        return Composition.construct(self, other)

    @staticmethod
    def is_zero(f:'Functionlike'):
        return isinstance(f, Scalar) and f.scalar == 0
    @staticmethod
    def is_one(f:'Functionlike'):
        return isinstance(f, Scalar) and f.scalar == 1
    @staticmethod
    def is_identity(f:'Functionlike'):
        return isinstance(f, Identity)
    
    @property
    def sort_val(self):
        return self.get_sortval()
    def get_sortval(self):
        return -1

    def simplify(self, iterations=10)->'Functionlike':
        old = new = self
        for r in range(iterations):
            new = old.apply_simplifications()
            if new == old:
                break
            old = new
        return new
    def apply_simplifications(self)->'Functionlike':
        return self

    @classmethod
    def merge_funcs(cls, funcs, reducer, iterations=10):
        for _ in range(iterations):
            unused = set(range(len(funcs)))
            new_funcs = []
            changed = False
            for i1, i2 in itertools.combinations(range(len(funcs)), 2):
                new = reducer(funcs[i1], funcs[i2])
                if new is not None:
                    changed = True
                    new_funcs.extend(new)
                    unused.remove(i1)
                    unused.remove(i2)
                    break
            new_funcs.extend(funcs[i] for i in unused)
            funcs = list(new_funcs)
            if not changed:
                break
        return funcs

    def __hash__(self):
        return id(self)

    def get_children(self):
        return []
    @property
    def children(self):
        return self.get_children()

    @classmethod
    def traverse(cls,
                 root,
                 traversal_order='depth',
                 visit_order='post',
                 node_test=None,
                 max_depth=None,
                 track_index=False
                 ):

        queue = collections.deque()
        if traversal_order == 'depth':
            queue_up = queue.appendleft
        else:
            queue_up = queue.appendleft

        queue_up(root if not track_index else ([], root))

        visit_queue = collections.deque()
        if visit_order == 'post':
            visit = visit_queue.appendleft
        else:
            visit = visit_queue.append

        if node_test is None:
            node_test = lambda n:True

        depth = 0
        while queue and (max_depth is None or depth < max_depth):
            cur = queue.popleft()
            if track_index:
                pos, cur = cur
            for n,c in enumerate(cur.children):
                if node_test(c):
                    if track_index:
                        queue_up((pos+[n], c))
                    else:
                        queue_up(c)
            if track_index:
                visit((pos, cur))
            else:
                visit(cur)
        return visit_queue

    def get_child(self, pos)->'Functionlike':
        raise NotImplementedError("{} is atomic".format(self))
    def replace_child(self, pos, new)->'Functionlike':
        raise NotImplementedError("{} is atomic".format(self))

    def tree_repr(self, sep="", indent=""):
        return "{}()".format(type(self).__name__)

class ElementaryFunction(Functionlike):
    """
    A _univariate_ function (though it can be threadable)
    that has known values and derivatives
    """
    __slots__ = ['idx']

    def __init__(self, *, idx=None):
        self.idx=idx

    @abc.abstractmethod
    def get_deriv(self) -> 'ElementaryFunction':
        ...

    def deriv(self, n=1, simplify=True):  # this can be overloaded but for now this is the easy way to bootstrap
        d = self.get_deriv()
        for _ in range(n - 1):
            d = d.get_deriv()
        if simplify:
            d = d.simplify()
        return d

    # @classmethod
    # def __getitem__(cls, item):  # to support Morse[0](r...) syntax
    #     return cls(idx=item)

    def __repr__(self):
        return "{}[{}]".format(
            type(self),
            self.idx
        )

    def idx_compatible(self, other):
        return other.idx is None or self.idx is None or other.idx == self.idx

class Variable(ElementaryFunction):
    def __init__(self, name, idx):
        super().__init__(idx=idx)
        self.name = name
    def eval(self, r: np.ndarray) -> 'np.ndarray':
        return r
    def get_compile_spec(self):
        return Abstract.Lambda(Abstract.Name(self.name))(Abstract.Name(self.name))
    def get_deriv(self) -> 'Functionlike':
        return Scalar(1)
    def simplify(self, iterations=10) ->'Functionlike':
        return self
    def __hash__(self):
        return hash((Variable, self.name, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Variable)
                and other.name == self.name
                and self.idx_compatible(other)
        )
    def __repr__(self):
        return self.name
    def tree_repr(self, sep="", indent=""):
        return "{}({})".format(type(self).__name__, self.name)

class ElementaryVaradic(ElementaryFunction):
    __slots__ = ['functions', 'idx']
    def __init__(self, *functions: ElementaryFunction, idx=None):
        if idx is None:
            for f in functions:
                if f.idx is not None:
                    idx = f.idx
                    break
        super().__init__(idx=idx)
        self.functions = functions

    sort_key = None
    def get_sortval(self):
        return self.sort_key + sum(f.sort_val for f in self.functions)
    def __hash__(self):
        return hash((type(self), self.functions, self.idx))
    def tree_equivalent(self, other):
        return (
                len(other.functions) == len(self.functions)
                and self.idx_compatible(other)
                and all(f1 == f2 for f1, f2 in zip(self.functions, other.functions))
        )
    def __eq__(self, other):
        return isinstance(other, type(self)) and self.tree_equivalent(other)
    def get_children(self):
        return self.functions

    @classmethod
    @abc.abstractmethod
    def get_repr(cls, fns)->str:
        ...
    def __repr__(self):
        return self.get_repr(self.functions)

    def tree_repr(self, sep="\n", indent=""):
        return "{name}({s}{terms}{s})".format(
            name=type(self).__name__,
            s=sep+indent,
            terms=(","+sep+indent).join(f.tree_repr(sep=sep, indent=indent+" ") for f in self.functions)
        )

    def get_child(self, pos):
        if isinstance(pos, (int, np.integer)):
            return self.functions[pos]
        else:
            fn = self
            for p in pos:
                fn = fn.get_child(p)
        return fn
    def replace_child(self, pos, new)->'ElementaryVaradic':
        if not isinstance(pos, (int, np.integer)):
            if len(pos) == 1:
                pos = pos[0]
            else:
                new = self.get_child(pos[0]).replace_child(pos[1:], new)
                pos = pos[0]

        fn = self.copy()
        fn.functions = fn.functions[:pos] + (new,) + fn.functions[pos+1:] # ez?
        return fn

class ElementarySummation(ElementaryVaradic):
    sort_key = 1000

    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.sum([f(r) for f in self.functions], axis=0)
    def get_compile_spec(self):
        var = self.get_compile_var()
        np = Abstract.Name('np')
        args = [f.get_compile_spec()(var) for f in self.functions]
        return Abstract.Lambda(var)(
            np.sum(np.array(args), axis=0)
        )
    def get_deriv(self) ->'ElementaryFunction':
        fns = self.functions
        return type(self)(*[f.deriv() for f in fns])
    @classmethod
    def get_repr(cls, fns):
        return "+".join(repr(x) for x in fns)

    @classmethod
    def merge_product(cls, f1, f2):
        if isinstance(f1, ElementaryProduct):
            scalars1 = [f.scalar for f in f1.functions if isinstance(f, Scalar)]
            terms1 = [f for f in f1.functions if not isinstance(f, Scalar)]
        else:
            scalars1 = [1]
            terms1 = [f1]

        if isinstance(f2, ElementaryProduct):
            scalars2 = [f.scalar for f in f2.functions if isinstance(f, Scalar)]
            terms2 = [f for f in f2.functions if not isinstance(f, Scalar)]
        else:
            scalars2 = [1]
            terms2 = [f2]

        if len(terms1) == len(terms2) and all(t1 == t2 for t1, t2 in zip(terms1, terms2)):
            return ElementaryProduct(Scalar(sum(scalars1)+sum(scalars2)), *terms1)

    @classmethod
    def reduce_pair(cls, f1, f2)->'Iterable[Functionlike]|bool':
        # implement possible reduction rules for sums
        mp = cls.merge_product(f1, f2)
        return [mp] if mp is not None else mp

    def apply_simplifications(self):
        simp_funs = [f.simplify() for f in sorted(self.functions, key=lambda f:f.sort_val)]
        newfs = [
            x
            for f in simp_funs
            for x in (f.functions if isinstance(f, ElementarySummation) else [f])
            if not self.is_zero(f)
        ]

        newfs = self.merge_funcs(newfs, self.reduce_pair)

        if len(newfs) == 0:
            return Scalar(0)
        elif len(newfs) == 1:
            return newfs[0]
        else:
            return type(self)(*sorted(newfs, key=lambda f: f.sort_val), idx=self.idx)

class ElementaryProduct(ElementaryVaradic):
    sort_key = 100

    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.prod([f(r) for f in self.functions], axis=0)
    def get_compile_spec(self):
        var = self.get_compile_var()
        np = Abstract.Name('np')
        args = [
            f.get_compile_spec()(var)
            for f in self.functions
        ]
        return Abstract.Lambda(var)(
            np.prod(np.array(args), axis=0)
        )
    def get_deriv(self) ->'ElementaryFunction':
        fns = self.functions
        prod = type(self)
        return ElementarySummation(*[
            prod(*[f if j!=i else f.deriv() for j,f in enumerate(fns)])
            for i in range(len(fns))
        ])

    @classmethod
    def reduce_pair(cls, f1, f2) -> 'Iterable[Functionlike]|bool':
        # implement possible reduction rules for products
        if isinstance(f1, Scalar) and isinstance(f2, Scalar):
            return [Scalar(f1.scalar*f2.scalar)]

    def apply_simplifications(self):
        simp_funs = [f.simplify() for f in sorted(self.functions, key=lambda f: f.sort_val)]
        for f in simp_funs:
            if self.is_zero(f):
                return Scalar(0)

        newfs = [
            x
            for f in simp_funs
            for x in (f.functions if isinstance(f, ElementaryProduct) else [f])
            if not self.is_one(f)
        ]

        newfs = self.merge_funcs(newfs, self.reduce_pair)
        
        if len(newfs) == 0:
            return Scalar(1)
        elif len(newfs) == 1:
            return newfs[0]
        else:
            return type(self)(*sorted(newfs, key=lambda f: f.sort_val), idx=self.idx)

    @classmethod
    def get_repr(cls, fns):
        return "*".join("({})".format(r) if isinstance(r, ElementarySummation) else repr(r) for r in fns)

class ElementaryComposition(ElementaryVaradic):
    sort_key = 0

    def eval(self, r:np.ndarray) ->'np.ndarray':
        return reduce(lambda r,f:f(r), reversed(self.functions), r)

    def get_compile_spec(self):
        var = self.get_compile_var()
        return Abstract.Lambda(var)(
            reduce(
                lambda r, f: f.get_compile_spec(r),
                reversed(self.functions),
                var
            )
        )

    def get_deriv(self) ->'ElementaryFunction':
        rev = list(reversed(self.functions))
        return ElementaryProduct(*[
            type(self)(f.get_deriv(), *rev[:i], idx=self.idx)
                if i > 0 else
            f.get_deriv()

            for i,f in enumerate(rev)
                if not isinstance(f, Variable)
        ])

    @classmethod
    def subs_identities(cls, f1, f2):
        idpos = cls.traverse(f1, track_index=True)
        changed = False
        for pos,node in idpos:
            if cls.is_identity(node):
                changed = True
                f1 = f1.replace_child(pos, f2)
        return changed, f1

    def apply_simplifications(self):
        simp_funs = [f.simplify() for f in self.functions]
        newf_point = 0
        for i,f in enumerate(simp_funs):
            if isinstance(f, Scalar):
                newf_point = i

        newfs = [
            x
            for f in simp_funs[newf_point:]
            for x in (f.functions if isinstance(f, ElementaryComposition) else [f])
            if not self.is_identity(f)
        ]

        # print("="*20)
        # for x in newfs:
        #     print(x.tree_repr())

        new_newfs = []
        curf = None
        for i in range(len(newfs)-1, -1, -1):# count backwards
            if curf is None:
                curf = newfs[i]
            else:
                changed, new = self.subs_identities(newfs[i], curf) # try to replace identities in newfs[i]
                if changed:
                    curf = new
                else:
                    new_newfs.append(curf)
                    curf = newfs[i]
        new_newfs.append(curf)
        newfs = list(reversed(new_newfs))
        # print("-"*10)
        # for x in newfs:
        #     print(x.tree_repr())


        if len(newfs) == 1:
            return newfs[0]
        elif len(newfs) == 0:
            return Identity()
        else:
            return type(self)(*newfs, idx=self.idx)

    @classmethod
    def get_repr(cls, fns):
        fmt_string = "{}({})"
        reprs = [repr(f) for f in fns]
        return reduce(
            lambda a, b:
            fmt_string.format(a, b)
            if "{}" not in a else
            a.replace("{}", "{0}").format(b if all(op not in b for op in ['+',"-","*"]) else "({})".format(b)),
            reprs
        )

class MultivariateFunction(Functionlike):
    """
    A multivariate function composed from elementary functions.
    This is a
    """

    def __init__(self, *functions:'ElementaryFunction|MultivariateFunction', indices=None):
        self.functions = functions
        self._inds = indices

    @classmethod
    def _reduce_indices(cls, terms):
        return {t.idx for t in terms if isinstance(t, ElementaryFunction) and t.idx is not None}
    @classmethod
    def construct_varivariate(cls, univariate, multivariate, terms, indices=None):
        il = cls._reduce_indices(terms)
        return (
            univariate(*terms, idx=indices[0] if indices is not None else indices)
            if len(il) < 2 and all(isinstance(t, ElementaryFunction) for t in terms) else
            multivariate(*terms, indices=indices)
        )

    @property
    def indices(self):
        if self._inds is None:
            terms = [
                [f.idx] if isinstance(f, ElementaryFunction) else f.indices for f in self.functions
                if not isinstance(f, ElementaryFunction) or f.idx is not None
            ]
            self._inds = np.unique(np.concatenate(terms)) if len(terms) > 0 else np.array([], dtype=int)
        return self._inds

    @property
    def ndim(self):
        return len(self.indices)

    @staticmethod
    def x_slice(x, idx):
        return ast.Subscript(
            value=x,
            slice=ast.Index(
                value=ast.Tuple(
                    elts=[ast.Constant(value=Ellipsis, kind=None), ast.Constant(value=idx, kind=None)],
                    ctx=ast.Load()
                )
            ),
            ctx=ast.Load()
        )

    @staticmethod
    def slice_lambda(expr, idx): # for taking slices in compiled statements
        return ast.Call(
            func=ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='x', annotation=None, type_comment=None)],
                    vararg=None,
                    kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                ),
                body=expr
            ),
            args=[
                ast.Subscript(
                    value=ast.Name(id='x', ctx=ast.Load()),
                    slice=ast.Index(
                        value=ast.Tuple(
                            elts=[ast.Constant(value=Ellipsis, kind=None), ast.Constant(value=idx, kind=None)],
                            ctx=ast.Load()
                            )
                    ),
                    ctx=ast.Load()
                )
            ],
            keywords=[]
        )

    @abc.abstractmethod
    def get_deriv(self, *counts) -> 'Functionlike':
        ...

    def deriv(self, *which, order=1, ndim=None, simplify=True) -> 'Functionlike':
        if len(which) == 0:
            ndim = self.ndim if ndim is None else ndim
            res = np.full((ndim,) * order, None)
            partitions = SymmetricGroupGenerator(ndim).get_terms(order)
            for i, pos in enumerate(itertools.combinations_with_replacement(range(ndim), r=order)):
                fun = self.get_deriv(*partitions[i])
                for p in itertools.permutations(pos):
                    res[p] = fun
            deriv = TensorFunction(res, symmetric=True, indices=self.indices)
        else:
            count_map = {k:v for k,v in zip(*np.unique(which, return_counts=True))}
            deriv = self.get_deriv(*(count_map.get(k, 0) for k in range(self.ndim)))

        if simplify:
            deriv = deriv.simplify()
        return deriv

    sort_key = None
    def get_sortval(self):
        return self.sort_key + sum(f.sort_val for f in self.functions)

    def apply_simplifications(self) ->'Functionlike':
        return type(self)(
            *(f.simplify() for f in self.functions),
            indices=self.indices
        )

    def __hash__(self):
        return hash((type(self), self.functions, self.indices))
    def tree_equivalent(self, other):
        return (
                len(other.functions) == len(self.functions)
                and (self.indices == other.indices).all()
                and all(f1 == f2 for f1, f2 in zip(self.functions, other.functions))
        )

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.tree_equivalent(other)

    def get_children(self):
        return self.functions

    # @classmethod
    # @abc.abstractmethod
    # def get_repr(cls, fns) -> str:
    #     ...
    # def __repr__(self):
    #     return self.get_repr(self.functions)

    def tree_repr(self, sep="\n", indent=""):
        return "{name}({s}{terms}{s})".format(
            name=type(self).__name__,
            s=sep + indent,
            terms=("," + sep + indent).join(f.tree_repr(sep=sep, indent=indent + " ") for f in self.functions)
        )

    def get_child(self, pos):
        if isinstance(pos, (int, np.integer)):
            return self.functions[pos]
        else:
            fn = self
            for p in pos:
                fn = fn.get_child(p)
        return fn
    def replace_child(self, pos, new) -> 'ElementaryVaradic':
        if not isinstance(pos, (int, np.integer)):
            if len(pos) == 1:
                pos = pos[0]
            else:
                new = self.get_child(pos[0]).replace_child(pos[1:], new)
                pos = pos[0]

        fn = self.copy()
        fn.functions = fn.functions[:pos] + (new,) + fn.functions[pos + 1:]  # ez?
        return fn

class TensorFunction(MultivariateFunction):
    """
    A tensor of functions
    """
    def __init__(self, functions:np.ndarray, symmetric=True, indices=None):
        functions = np.asanyarray(functions, dtype=object)
        self.symmetric = symmetric
        super().__init__(*functions.flat, indices=indices)
        self.functions = functions
    def _get_res_array(self, v)->np.ndarray:
        if isinstance(v, np.ndarray):
            res = np.empty(self.functions.shape + v.shape, dtype=v.dtype)
        elif isinstance(v, str):
            res = np.empty(self.functions.shape, dtype=object) # someday maybe I'll treat as strings...?
        elif isinstance(v, (int, np.integer)):
            res = np.empty(self.functions.shape, dtype=int)
        elif isinstance(v, (float, np.floating)):
            res = np.empty(self.functions.shape, dtype=float)
        else:
            res = np.empty(self.functions.shape, dtype=object)
        return res
    def apply_function(self, fn, res_builder=None)->np.ndarray:
        flat = self.functions.flat
        if self.symmetric:
            _cache = {}
        if res_builder is None:
            res_builder = self._get_res_array
        res = None
        for f in flat:
            idx = np.unravel_index(flat.index - 1, self.functions.shape)
            if not self.symmetric:
                v = fn(f)
                if res is None:
                    res = res_builder(v)
                res[idx] = v
            else:
                key = tuple(np.sort(idx))
                if key not in _cache:
                    _cache[key] = fn(f)
                if res is None:
                    res = res_builder(_cache[key])
                res[idx] = _cache[key]
        return res
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return self.apply_function(
            lambda f:f(r[..., f.idx] if f.idx is not None else r) if isinstance(f, ElementaryFunction) else f(r)
        )
    def get_compile_spec(self) ->'ast.Lambda':
        var = self.get_compile_var()
        var_2 = Abstract.Name("_")
        spec_calls = [
            f.get_compile_spec()(
                args=[
                    self.x_slice(ast.Name(id=var, ctx=ast.Load()), f.idx)
                    if (isinstance(f, ElementaryFunction) and f.idx is not None) else
                    ast.Name(var, ctx=ast.Load())
                ],
                keywords=[]
            )

            for f in self.functions.flat
        ]
        # call with appropriate slicing behavior
        spec_array = self.ast_nparray(*spec_calls)

        return self.lambda_wrap(
            var,
            ast.Call( # just a reshape...
                func=self.lambda_wrap(
                    var_2,
                    self.ast_np('reshape',
                                ast.Name(id=var_2, ctx=ast.Load()),
                                ast.BinOp(
                                    left=ast.Tuple(elts=[ast.Constant(value=s, kind=None) for s in self.functions.shape], ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Subscript(
                                        value=self.ast_attr(var_2, 'shape'),
                                        slice=ast.Slice(lower=ast.Constant(value=1, kind=None), upper=None, step=None), ctx=ast.Load()
                                    )
                                )
                        )
                ),
                args=[spec_array],
                keywords=[]
            )
        )
    def get_deriv(self, *counts)->'TensorFunction':
        return TensorFunction(self.apply_function(lambda f:f.get_deriv(*counts)), indices=self.indices)
    def get_sortval(self):
        return self.sort_key + sum(f.sort_val for f in self.functions.flat)
    def apply_simplifications(self) ->'Functionlike':
        return type(self)(self.apply_function(lambda f:f.simplify()), indices=self.indices)
    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.functions)

    @classmethod
    def format_repr_array(cls, arr, ilevel=0, brackets="[]", sep=",\n", indent=" "):
        # # non-recursive depth-first walk of the array tree rather than
        # # itertools based or arr.flat based walk since both of those require
        # # explicit tracking of indices
        # bl, br = brackets
        # queue = collections.deque()
        # queue.append(arr)
        # lines = []
        # while queue:
        #     cur = queue.pop()
        #     if isinstance(cur[0], np.ndarray):
        #         queue.extend(cur)
        #         lines.append(bl)
        #         ilevel += 1
        #     else:
        #         lines.append(indent+bl)
        #         lines.append((sep+indent).join(cur))
        #         lines.append(indent+br)
        #         ilevel -= 1
        # return "\n".join(lines)

        ## Meh. Recursion is easier.
        lines = []
        lines.append((ilevel*indent)+brackets[0])
        if isinstance(arr[0], np.ndarray):
            for row in arr:
                lines.append(cls.format_repr_array(row, ilevel=ilevel+1, brackets=brackets, sep=sep, indent=indent))
        else:
            lines.append((ilevel*indent)+sep.join(arr))
        lines.append((ilevel*indent)+brackets[1])
        return "\n".join(lines)

    def tree_equivalent(self, other):
        return (
                len(other.functions) == len(self.functions)
                and (self.indices == other.indices).all()
                and (self.functions == other.functions).all()
        )
    def tree_repr(self, sep="\n", indent=""):
        return "{name}({terms})".format(
            name=type(self).__name__,
            terms=self.format_repr_array(self.apply_function(lambda f:f.tree_repr(sep=sep, indent=indent + " ")))
        )
    def copy(self)->'TensorFunction':
        cp = super().copy() #type: TensorFunction
        cp.functions = cp.functions.copy()
        return cp

    def get_children(self)->'Iterable[Functionlike]':
        return self.functions.flat
    def get_child(self, pos)->'Functionlike':
        if isinstance(pos, (int, np.integer)):
            pos = np.unravel_index(pos, self.functions.shape)
            return self.functions[pos]
        else:
            fn = self
            for p in pos:
                fn = fn.get_child(p)
        return fn
    def replace_child(self, pos, new) -> 'TensorFunction':
        if not isinstance(pos, (int, np.integer)):
            if len(pos) == 1:
                pos = pos[0]
            else:
                new = self.get_child(pos[0]).replace_child(pos[1:], new)
                pos = pos[0]

        fn = self.copy()
        fn.functions[pos] = new
        return fn

class Summation(MultivariateFunction):
    """
    A summation of 1D functions to support testing derivs
    """

    def eval(self, r: np.ndarray) -> 'np.ndarray':
        vals = [f(r[..., f.idx] if f.idx is not None else r) if isinstance(f, ElementaryFunction) else f(r) for f in self.functions]
        return np.sum(vals, axis=0)
    def get_compile_spec(self) -> 'ast.Lambda':
        var = self.get_compile_var()
        args = [
            ast.Call(
                func=f.get_compile_spec(),
                args=[
                    self.x_slice(ast.Name(id=var, ctx=ast.Load()), f.idx)
                    if (isinstance(f, ElementaryFunction) and f.idx is not None) else
                    ast.Name(var, ctx=ast.Load())
                ],
                keywords=[]
            )

            for f in self.functions
        ]
        return self.lambda_wrap(
            var,
            self.ast_np('sum',
                        self.ast_nparray(*args),
                        axis=ast.Constant(value=0, kind=None)
                        )
        )

    @classmethod
    def construct(cls, *terms, indices=None):
        terms = [
            x
            for t in terms
            for x in (
                t.functions
                if isinstance(t, (Summation, ElementarySummation))
                else [t]
            )
        ]
        terms = [t for t in terms if not (isinstance(t, Scalar) and t.scalar == 0)]
        if len(terms) == 0:
            return Scalar(0)
        return cls.construct_varivariate(ElementarySummation, cls, terms, indices=indices)

    def get_deriv(self, *counts)->'MultivariateFunction':
        needs_der_pos = {i for i,c in enumerate(counts) if c > 0}
        if len(needs_der_pos) == 1:
            idx = list(needs_der_pos)[0]
        else:
            idx = -1
        return type(self)(
            *(
                f.deriv(counts[idx])
                    if isinstance(f, ElementaryFunction) else
                f.get_deriv(*counts)
                for i, f in enumerate(self.functions)

                if not isinstance(f, ElementaryFunction) or
                   len(needs_der_pos) == 1 and (f.idx is None or f.idx in needs_der_pos)
            ),
            indices=self.indices
        )

    sort_key = 1000

    @classmethod
    def merge_product(cls, f1, f2):
        if isinstance(f1, (Product, ElementaryProduct)):
            scalars1 = [f.scalar for f in f1.functions if isinstance(f, Scalar)]
            terms1 = [f for f in f1.functions if not isinstance(f, Scalar)]
        else:
            scalars1 = [1]
            terms1 = [f1]

        if isinstance(f2,  (Product, ElementaryProduct)):
            scalars2 = [f.scalar for f in f2.functions if isinstance(f, Scalar)]
            terms2 = [f for f in f2.functions if not isinstance(f, Scalar)]
        else:
            scalars2 = [1]
            terms2 = [f2]

        if len(terms1) == len(terms2) and all(t1 == t2 for t1, t2 in zip(terms1, terms2)):
            return type(f1)(Scalar(sum(scalars1) + sum(scalars2)), *terms1)

    @classmethod
    def reduce_pair(cls, f1, f2) -> 'Iterable[Functionlike]|bool':
        # implement possible reduction rules for sums
        mp = cls.merge_product(f1, f2)
        return [mp] if mp is not None else mp

    def apply_simplifications(self):
        simp_funs = [f.simplify() for f in sorted(self.functions, key=lambda f: f.sort_val)]
        newfs = [
            x
            for f in simp_funs
            for x in (f.functions if isinstance(f, (ElementarySummation, Summation)) else [f])
            if not self.is_zero(f)
        ]

        # newfs = self.merge_funcs(newfs, self.reduce_pair)

        if len(newfs) == 0:
            return Scalar(0)
        elif len(newfs) == 1:
            return newfs[0]
        else:
            return type(self)(*sorted(newfs, key=lambda f: f.sort_val), indices=self.indices)

    def __repr__(self):
        return ElementarySummation.get_repr(self.functions)

class Product(MultivariateFunction):
    """
    A summation of 1D functions to support testing derivs
    """

    def eval(self, r: np.ndarray) -> 'np.ndarray':
        vals = [f(r[..., f.idx] if f.idx is not None else r) if isinstance(f, ElementaryFunction) else f(r) for f in self.functions]
        return np.product(vals, axis=0)
    def get_compile_spec(self) ->'ast.Lambda':
        var = self.get_compile_var()
        args = [
            ast.Call(
                func=f.get_compile_spec(),
                args=[
                    self.x_slice(ast.Name(id=var, ctx=ast.Load()), f.idx)
                        if (isinstance(f, ElementaryFunction) and f.idx is not None) else
                    ast.Name(var, ctx=ast.Load())
                ],
                keywords=[]
            ) for f in self.functions
        ]
        return self.lambda_wrap(
            var,
            self.ast_np('prod',
                        self.ast_nparray(*args),
                        axis=ast.Constant(value=0, kind=None)
                        )
        )

    @classmethod
    def construct(cls, *terms, indices=None):
        terms = [
            x
            for t in terms
            for x in (
                t.functions
                if isinstance(t, (Product, ElementaryProduct))
                else [t]
            )
        ]
        for t in terms:
            if isinstance(t, Scalar) and t.scalar == 0:
                return Scalar(0)
        terms = [t for t in terms if not (isinstance(t, Scalar) and t.scalar == 1)]
        if len(terms) == 0:
            return Scalar(1)
        return cls.construct_varivariate(ElementaryProduct, cls, terms, indices=indices)

    def get_deriv(self, *counts):
        needs_der_pos = {i for i, c in enumerate(counts) if c > 0}
        if len(needs_der_pos) == 1:
            idx = list(needs_der_pos)[0]
        else:
            idx = -1
        return Summation.construct(
            *(
                type(self)(
                    *(
                        (
                            f.deriv(counts[idx])
                                if isinstance(f, ElementaryFunction) else
                            f.get_deriv(*counts)
                        )

                        if j == i else

                        f

                        for j, f in enumerate(self.functions)
                    ),
                    indices=self.indices
                )

                for i in range(len(self.functions))
                if not isinstance(self.functions[i], ElementaryFunction) or
                   len(needs_der_pos) == 1 and (self.functions[i].idx is None or self.functions[i].idx in needs_der_pos)
            ),
            indices=self.indices
        )

    sort_key = 100

    def apply_simplifications(self):
        simp_funs = [f.simplify() for f in self.functions]
        for f in simp_funs:
            if isinstance(f, Scalar) and f.scalar == 0:
                return Scalar(0)

        newfs = [
            f for f in simp_funs
            if not (isinstance(f, Scalar) and f.scalar == 1)
        ]
        if len(newfs) == 0:
            return Scalar(1)
        return type(self)(*newfs, indices=self.indices)

    def __repr__(self):
        return ElementaryProduct.get_repr(self.functions)

class Composition(MultivariateFunction):
    """
    A composition of multivariate functions that
    uses the chain rule for derivatives
    """

    def eval(self, r: np.ndarray) -> 'np.ndarray':
        return reduce(lambda r,f: f(r[..., f.idx] if f.idx is not None else r), reversed(self.functions), r)
    def get_compile_spec(self) ->'ast.Lambda':
        var = self.get_compile_var()
        return self.lambda_wrap(
            var,
            reduce(
                lambda r, f: ast.Call(
                    func=f.get_compile_spec(),
                    args=[
                        self.x_slice(r, f.idx)
                        if (isinstance(f, ElementaryFunction) and f.idx is not None) else
                        r
                    ],
                    keywords=[]
                ),
                reversed(self.functions),
                ast.Name(id=var, ctx=ast.Load())
            )
        )

    @classmethod
    def construct(cls, *terms, indices=None):
        terms = [
            x
            for t in terms
            for x in (
                t.functions
                if isinstance(t, (Composition, ElementaryComposition))
                else [t]
            )
        ]
        sel = 0
        for i,t in enumerate(terms):
            if isinstance(t, Scalar):
                sel = i
        terms = terms[sel:]
        return cls.construct_varivariate(ElementaryComposition, cls, terms, indices=indices)

    def apply_simplifications(self):
        simp_funs = [f.simplify() for f in self.functions]
        newf_point = 0
        for i,f in enumerate(simp_funs):
            if isinstance(f, Scalar):
                newf_point = i

        newfs = [f for f in simp_funs[newf_point:] if not isinstance(f, Identity)]
        if len(newfs) == 1:
            return newfs[0]
        elif len(newfs) == 0:
            return Identity()
        else:
          return type(self)(*newfs, indices=self.indices)

    # def get_deriv(self) ->'ElementaryFunction':
    #     return ElementaryProduct(*[
    #         # this could be evaluated more efficiently, admittedly...
    #         f.get_deriv() *type(self)(*self.functions[:i], idx=self.idx) if i > 0 else f.get_deriv()
    #         for i,f in enumerate(self.functions)
    #     ])

    # def single_deriv(self, f, arg, ndim, coord):
    #     if isinstance(f, ElementaryFunction):
    #         ...
    #     else:
    #         grad = [None]*ndim
    #         for i in range(ndim):
    #             counts = [0] * ndim
    #             counts[i] = 1
    #             grad[i] = f.get_deriv(*counts)
    #
    #         grad = f.get_deriv(c) for c in co

    def get_deriv(self, *counts):
        # do we assume compositions are always of a single arg
        # or do we allow for multiarg compositions???
        #   --> I think by design we have single arg and then we include multiarg later?

        needs_der_pos = {i for i, c in enumerate(counts) if c > 0}
        nonzero = all(
            not isinstance(f, ElementaryFunction) or
                len(needs_der_pos) == 1 and f.idx in needs_der_pos
            for f in self.functions
        )
        if nonzero:
            c = list(needs_der_pos)[0]
            return Product.construct(*(
                (
                    (
                        f.get_deriv(*counts)
                        if not isinstance(f, ElementaryFunction) else
                        f.get_deriv(c)
                    ) * self.construct(*self.functions[:i])
                    if i > 0 else f.get_deriv()
                )
                for i,f in enumerate(self.functions)
            ), indices=self.indices)
        else:
            return Scalar(0)

    def __repr__(self):
        return ElementaryComposition.get_repr(self.functions)

class Scalar(ElementaryFunction):
    """
    Broadcasts a constant value
    """
    __slots__ = ['scalar', 'idx']
    def __init__(self, scalar, *, idx=None):
        if not isinstance(scalar, Functionlike):
            idx = None
        super().__init__(idx=idx)
        self.scalar = scalar
    def eval(self, r: np.ndarray) -> 'np.ndarray':
        return np.broadcast_to(np.full(1, self.scalar, dtype=r.dtype), r.shape)
    def get_compile_spec(self) ->'ast.Lambda':
        if isinstance(self.scalar, Functionlike):
            return self.scalar.get_compile_spec()
        else:
            var = self.get_compile_var()
            return self.lambda_wrap(
                var,
                self.ast_np('full',
                            self.ast_attrr(self.ast_var(var), 'shape'),
                            ast.Constant(value=self.scalar, kind=None)
                            )
            )
    def get_deriv(self) -> 'ElementaryFunction':
        return type(self)(0, idx=self.idx)
    def deriv(self, n=1, *, simplify=True):
        return type(self)(0, idx=self.idx)
    def get_sortval(self):
        return (self.scalar.sort_val if isinstance(self.scalar, Functionlike) else self.scalar)
    def __hash__(self):
        return hash((Scalar, self.scalar, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Scalar)
                and self.idx_compatible(other)
                and self.scalar == other.scalar
        )
    def __repr__(self):
        return repr(self.scalar)
    def tree_repr(self, sep="\n", indent=""):
        return "{}({})".format(type(self).__name__, self.scalar)

class Identity(ElementaryFunction):
    """
    Identity function for compositions
    """
    def eval(self, r: np.ndarray) -> 'np.ndarray':
        return r
    def get_compile_spec(self) ->'ast.Lambda':
        return ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='x', annotation=None, type_comment=None)],
                    vararg=None,
                    kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                ),
                body=ast.Name(id='x', ctx=ast.Load())
            )
    def get_deriv(self) -> 'ElementaryFunction':
        return Scalar(1)
    def deriv(self, n=1, simplify=True):
        return Scalar(1) if n == 1 else Scalar(0)
    def simplify(self, iterations=10) ->'Functionlike':
        return self
    def __hash__(self):
        return hash((Identity, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Identity)
                and self.idx_compatible(other)
        )
    def get_sortval(self):
        return 100
    def __repr__(self):
        return "{}"

class Power(ElementaryFunction):
    __slots__ = ['power', 'idx']
    def __init__(self, power, *, idx=None):
        super().__init__(idx=idx)
        self.power = power
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return r**self.power
    def get_compile_spec(self) ->'ast.Lambda':
        power = self.power.get_compile_spec() if isinstance(self.power, Functionlike) else ast.Constant(value=self.power, kind=None)
        return ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='x', annotation=None, type_comment=None)],
                    vararg=None,
                    kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                ),
                body=ast.BinOp(left=ast.Name(id='x', ctx=ast.Load()), op=ast.Pow(), right=power)
            )
    def get_deriv(self) -> 'ElementaryFunction':
        return self.power * type(self)(self.power-1, idx=self.idx)
    def __hash__(self):
        return hash((Power, self.power, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Power)
                and self.idx_compatible(other)
                and self.power == other.power
        )
    sort_key = 20
    def get_sortval(self):
        return self.sort_key + (self.power.sort_val if isinstance(self.power, Functionlike) else self.power)
    def apply_simplifications(self) ->'Functionlike':
        power = self.power
        if isinstance(power, Scalar):
            power = power.scalar
        if isinstance(power, (int, np.integer, float, np.floating)):
            if power == 1:
                return Identity()
            elif power == 0:
                return Scalar(1)
        return self
    def __repr__(self):
        return "{{}}^{}".format(self.power)
    def tree_repr(self, sep="\n", indent=""):
        return "{}({})".format(type(self).__name__, self.power.tree_repr(sep=sep, indent=indent+" ") if isinstance(self.power, Functionlike) else self.power)

class Exponent(ElementaryFunction):
    __slots__ = ['base', 'idx']
    def __init__(self, base, *, idx=None):
        super().__init__(idx=idx)
        self.base = base
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return self.base**r
    def get_compile_spec(self) ->'ast.Lambda':
        base = self.base.get_compile_spec() if isinstance(self.base, Functionlike) else ast.Constant(value=self.base, kind=None)
        return ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='x', annotation=None, type_comment=None)],
                    vararg=None,
                    kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                ),
                body=ast.BinOp(left=base, op=ast.Pow(), right=ast.Name(id='x', ctx=ast.Load()))
            )
    def get_deriv(self) -> 'ElementaryFunction':
        return np.log(self.base) * self
    def __hash__(self):
        return hash((Exponent, self.base, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Exponent)
                and self.idx_compatible(other)
                and self.base == other.base
        )
    sort_key = 50
    def get_sortval(self):
        return self.sort_key + (self.base.sort_val if isinstance(self.base, Functionlike) else self.base)
    def __repr__(self):
        return "{}^{{}}".format(self.base)
    def tree_repr(self, sep="\n", indent=""):
        return "{}({})".format(type(self).__name__, self.base.tree_repr(sep=sep, indent=indent+" ") if isinstance(self.base, Functionlike) else self.base)
class Exp(Exponent):
    __slots__ = ['idx']
    def __init__(self, *, idx=None):
        super().__init__(np.e, idx=idx)
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.exp(r)
    def get_compile_spec(self) ->'ast.Attribute':
        return self.ast_attr('np', 'exp')
    def get_deriv(self) -> 'ElementaryFunction':
        return self
    def __repr__(self):
        return "exp{}"
    def tree_repr(self, sep="\n", indent=""):
        return "{}()".format(type(self).__name__)

class Logarithm(ElementaryFunction):
    __slots__ = ['base', 'idx']
    def __init__(self, base, *, idx=None):
        super().__init__(idx=idx)
        self.base = base
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.log(r)/np.log(self.base)
    def get_deriv(self) -> 'ElementaryFunction':
        return 1/(np.log(self.base)*self)
    def __hash__(self):
        return hash((Logarithm, self.base, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Logarithm)
                and self.idx_compatible(other)
                and self.base == other.base
        )
    sort_key = 60
    def get_sortval(self):
        return self.sort_key + (self.base.sort_val if isinstance(self.base, Functionlike) else self.base)
    def __repr__(self):
        return "log_{}{{}}".format(self.base)
    def tree_repr(self, sep="\n", indent=""):
        return "{}({})".format(type(self).__name__, self.base.tree_repr(sep=sep, indent=indent+" ") if isinstance(self.base, Functionlike) else self.base)
class Ln(Logarithm):
    __slots__ = ['idx']
    def __init__(self, *, idx=None):
        super().__init__(np.e, idx=idx)
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.log(r)
    def get_compile_spec(self) ->'ast.Attribute':
        return self.ast_attr('np', 'log')
    def get_deriv(self) -> 'ElementaryFunction':
        return 1/self
    def __repr__(self):
        return "ln{}"
    def tree_repr(self, sep="\n", indent=""):
        return "{}()".format(type(self).__name__)

class Sin(ElementaryFunction):
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.sin(r)
    def get_deriv(self) -> 'ElementaryFunction':
        return Cos()
    def get_compile_spec(self) ->'ast.Attribute':
        return self.ast_attr('np', 'sin')
    def __hash__(self):
        return hash((Sin, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Sin)
                and self.idx_compatible(other)
        )
    def __repr__(self):
        return "sin{}"
class Cos(ElementaryFunction):
    def eval(self, r:np.ndarray) ->'np.ndarray':
        return np.cos(r)
    def get_compile_spec(self) ->'ast.Attribute':
        return self.ast_attr('np', 'cos')
    def get_deriv(self) -> 'ElementaryFunction':
        return -Sin()
    def __hash__(self):
        return hash((Cos, self.idx))
    def __eq__(self, other):
        return (
                isinstance(other, Cos)
                and self.idx_compatible(other)
        )
    def __repr__(self):
        return "cos{}"

class CompoundFunction(ElementaryFunction):
    def __init__(self, *, idx=None):
        super().__init__(idx=idx)
        self._expr = None
    @abc.abstractmethod
    def get_expression(self) -> 'ElementaryFunction':
        ...
    @property
    def expression(self):
        if self._expr is None:
            self._expr = self.get_expression()
        return self._expr
    def __eq__(self, other):
        return self.expression == other
    def get_deriv(self) -> 'ElementaryFunction':
        return self.expression.get_deriv()
    def get_compile_spec(self) -> 'ast.AST':
        return self.expression.get_compile_spec()
    def get_sortval(self):
        return self.expression.sort_val
class Morse(CompoundFunction):
    def __init__(self, *, de=1, a=1, re=0, idx=None):
        self.de = de
        self.a = a
        self.re = re
        super().__init__(idx=idx)
    def get_expression(self):
        de = self.de if isinstance(self.de, Scalar) else Scalar(self.de)
        a = self.a if isinstance(self.a, Scalar) else Scalar(self.a)
        re = self.re if isinstance(self.re, Scalar) else Scalar(self.re)
        r = Identity(idx=self.idx)
        return de*(Scalar(1)-Exp()(-a*(r-re)))**2
    def eval(self, r: np.ndarray) -> 'np.ndarray':
        return self.de*(1-np.exp(-self.a*(r-self.re )))**2
    def __repr__(self):
        return "morse(de={},a={},re={})".format(self.de, self.a, self.re)
    def tree_repr(self, sep="\n", indent=""):
        return "{}(de={},a={},re={})".format(type(self).__name__, self.de, self.a, self.re)

class Symbols:

    def __init__(self, *vars):
        if len(vars) == 1 and isinstance(vars[0], str):
            vars = vars[0]
            if " " in vars:
                vars = vars.split()
        self.vars = [Variable(v, i) if not isinstance(v, Variable) else v for i, v in enumerate(vars)]
        self.__dict__.update({v.name: v for v in self.vars})
    @classmethod
    def scalar(cls, v):
        return Scalar(v)

    @classmethod
    def log(cls, v, base=None):
        return Exp()(v) if base is None else Logarithm(base)(v)

    @classmethod
    def exp(cls, v, base=None):
        return Exp()(v) if base is None else Exponent(base)(v)

    @classmethod
    def cos(cls, x):
        return Cos()(x)

    @classmethod
    def sin(cls, x):
        return Sin()(x)

    @classmethod
    def morse(cls, r, de=1, a=1, re=0):
        return Morse(de=de, a=a, re=re)(r)