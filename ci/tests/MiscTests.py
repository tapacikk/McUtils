
from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Misc import *
import numpy
# import ast, astunparse, numpy

class MiscTests(TestCase):

    @debugTest
    def test_Symbolics(self):
        x, y, z, some = Abstract.vars('x', 'y', 'z', 'some')
        lexpr = Abstract.Lambda(x, *y, some=1, **z)(
            x*some
        )

        # print(ast.dump(lexpr.to_eval_expr()))

        lfun = lexpr.compile()

        self.assertEquals([1, 2, 3]*3, lfun([1, 2, 3], this=1, has=0, some=3, effect=4))

        x, np = Abstract.vars('x', 'np')
        npexpr = Abstract.Lambda(x)(
            np.array(x)[..., 0] + 1
        )

        # print(
        #     ast.dump(
        #         npexpr.to_eval_expr()
        #     )
        # )

        self.assertTrue(
            numpy.all(
                npexpr.compile({'np':numpy})([[1], [2], [3]])
                == numpy.array([[1], [2], [3]])[..., 0] + 1
            )
        )


