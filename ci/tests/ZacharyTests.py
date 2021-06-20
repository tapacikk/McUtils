
from Peeves.TestUtils import *
from McUtils.Zachary import *
from McUtils.Zachary.Taylor.ZachLib import *
from McUtils.Plots import *
from unittest import TestCase
import sys, h5py, math, numpy as np, itertools

class FiniteDifferenceTests(TestCase):

    #region setup

    def setUp(self):
        self.save_data = TestManager.data_gen_tests

    # @validationTest
    # def test_finite_difference_2d(self):
    #     grid_1 = 4*np.math.pi / 100 * np.arange(100) + 2*np.math.pi
    #     grid = np.meshgrid(grid_1, grid_1)
    #     self.assertAlmostEqual(1, 0)#round(np.linalg.norm(dtest - sin_d3_vals), 4), 0)

    def get_error(self, ref_vals, vals):
        err = np.abs(vals - ref_vals)
        cum_err = np.linalg.norm(err.flatten())
        max_err = np.max(err.flatten())
        mean_err = np.average(err.flatten())
        return err, cum_err, max_err, mean_err

    def plot_err(self, grid, ref_vals, vals, errs):
        if grid.ndim == 1:
            Plot(grid, ref_vals, figure=Plot(grid, vals))
            Plot(grid, errs[0]).show()
        elif grid.ndim == 3:
            g = (grid[:, :, 0], grid[:, :, 1])
            gg = GraphicsGrid(nrows=2, ncols=2)
            gg[0, 0] = ContourPlot(*g, ref_vals, figure=gg[0, 0])
            gg[1, 0] = ContourPlot(*g, vals, figure=gg[1, 0])
            gg[1, 1] = ContourPlot(*g, errs[0], figure=gg[1, 1])
            gg[0, 0].tight_layout()
            gg.show()

    def print_error(self, n, order, errs):
        print(
            "Order: {}.{}\nCumulative Error: {}\nMax Error: {}\nMean Error: {}".format(n, order, *errs[1:])
        )

    #endregion

    #region FD inputs
    @validationTest
    def test_stirs(self):
        stir = StirlingS1(8)
        ans = np.array([
            [1,  0,    0,     0,     0,   0,    0,  0],
            [0,  1,    0,     0,     0,   0,    0,  0],
            [0, -1,    1,     0,     0,   0,    0,  0],
            [0,  2,   -3,     1,     0,   0,    0,  0],
            [0, -6,    11,   -6,     1,   0,    0,  0],
            [0,  24,  -50,    35,   -10,  1,    0,  0],
            [0, -120,  274,  -225,   85, -15,   1,  0],
            [0,  720, -1764,  1624, -735, 175, -21, 1]
        ])
        # print(stir, file=sys.stderr)
        self.assertAlmostEqual(np.round(np.linalg.norm(stir-ans), 6), 0.)

    @validationTest
    def test_bin_gs(self):
        gbin = GammaBinomial(7/2, 8)
        ans = np.array([1., 3.5, 4.375, 2.1875, 0.273438, -0.0273438, 0.00683594, -0.00244141])
        # print(gbin, file=sys.stderr)
        self.assertAlmostEqual(np.round(np.linalg.norm(gbin-ans), 5), 0.)
    @validationTest
    def test_bins(self):
        gbin = Binomial(8)
        ans = np.array([
            [1, 0, 0,  0,  0,  0,  0, 0],
            [1, 1, 0,  0,  0,  0,  0, 0],
            [1, 2, 1,  0,  0,  0,  0, 0],
            [1, 3, 3,  1,  0,  0,  0, 0],
            [1, 4, 6,  4,  1,  0,  0, 0],
            [1, 5, 10, 10, 5,  1,  0, 0],
            [1, 6, 15, 20, 15, 6,  1, 0],
            [1, 7, 21, 35, 35, 21, 7, 1]
        ])
        # print(gbin, file=sys.stderr)
        self.assertAlmostEqual(np.sum(gbin-ans), 0.)
    @validationTest
    def test_facs(self):
        facs = Factorial(8)
        ans = np.array([1, 1, 2, 6, 24, 120, 720, 5040])
        # print(facs, file=sys.stderr)
        self.assertAlmostEqual(np.linalg.norm(facs-ans), 0.)
    @validationTest
    def test_fd_weights(self):
        coeffs = RegularGridFiniteDifference.get_weights(3, 7/2, 7)
        ans = [ -0.0192708, 0.259896, -2.02969, 4.92448, -4.92448, 2.02969, -0.259896, 0.0192708 ]
        coeffs2 = RegularGridFiniteDifference.get_weights(3, 0, 7)
        ans2 = [
            -8.058333333333334, 42.53333333333333, -98.225,
            129.66666666666666, -106.04166666666667, 53.6,
            -15.408333333333333, 1.9333333333333333
        ]
        # print(coeffs2, file=sys.stderr)
        r1 = np.round(np.linalg.norm(coeffs-ans), 5)
        r2 = np.round(np.linalg.norm(coeffs2-ans2), 10)
        self.assertAlmostEqual(r1 + r2, 0.)
    @validationTest
    def test_uneven_weights(self):
        import numpy as np
        weights = IrregularGridFiniteDifference.get_weights
        uweights =  [
            weights(2, 0, np.array([-2, -1, 0, 1, 2])),
            weights(1, 0, np.array([-3/2, -1/2, 1/2, 3/2])),
            weights(1, 1, np.array([-3, -2, -1, 0, 1]))
            #weights(3, 1/2, np.array([0, 1/3, 1, 2, 7/2, 6]))
            ]
        eweights = RegularGridFiniteDifference.get_weights
        targ_weights = [
            eweights(2, 2,   4),
            eweights(1, 3/2, 3),
            eweights(1, 4,   4)
        ]
        passed = True
        for e, w in zip(targ_weights, uweights):
            norm = np.linalg.norm(e-w[-1])
            if norm > .000001:
                passed = False
                print((norm, e, w[-1]), file=sys.stderr)
        self.assertIs(passed, True)

    #endregion

    #region FD
    # @dataGenTest
    @validationTest
    def test_finite_difference(self):
        sin_grid = np.arange(0, 1, .001)
        sin_vals = np.sin(sin_grid)
        sin_d3_vals = -np.cos(sin_grid)

        if self.save_data:
            f = h5py.File(TestManager.test_data('fd_data_1D.hdf5'),'w')
            f.create_dataset("vals", data=sin_vals)
            f.create_dataset("ref", data=sin_d3_vals)

        print_error = False
        plot_error = False

        ref_vals = np.cos(sin_grid)
        for ord in range(2, 10):
            n = 1
            sten = ord
            # core_dvals = sin_d_vals[math.floor(sten/2):-math.floor(sten/2)]
            vals = finite_difference(sin_grid, sin_vals, n, stencil=sten, end_point_accuracy=0,
                                        mode="sparse"
                                        )

            errs = self.get_error(ref_vals, vals)
            if plot_error:
                self.plot_err(sin_grid, ref_vals, vals, errs)
            if print_error:
                self.print_error(n, ord, errs)
            self.assertLess(errs[1], .1/ord)

        ref_vals = -np.sin(sin_grid)
        for ord in range(3, 10):
            n=2
            sten = ord
            vals = finite_difference(sin_grid, sin_vals, n, stencil=sten, end_point_accuracy=0,
                                        mode="dense"
                                        )

            errs = self.get_error(ref_vals, vals)
            if plot_error:
                self.plot_err(sin_grid, ref_vals, vals, errs)
            if print_error:
                self.print_error(n, ord, errs)
            self.assertLess(errs[1], .05 / ord)

        ref_vals = sin_d3_vals
        for ord in range(4, 10):
            n = 3
            sten = ord
            vals = finite_difference(sin_grid, sin_vals, n, ord, stencil=sten, end_point_accuracy=2,
                                        mode="sparse"
                                        )

            errs = self.get_error(ref_vals, vals)
            if plot_error:
                self.plot_err(sin_grid, ref_vals, vals, errs)
            if print_error:
                self.print_error(n, ord, errs)
            self.assertLess(errs[1], .05 / ord)

    @validationTest
    def test_FD2D(self):

        print_error = False
        plot_error = False

        x_grid = np.arange(0, 1, .01, dtype=np.longdouble)
        y_grid = np.arange(0, 1, .001, dtype=np.longdouble)

        sin_x_vals = np.sin(x_grid); sin_y_vals =  np.sin(y_grid)
        cos_x_vals = np.cos(x_grid); cos_y_vals =  np.cos(y_grid)
        vals_2D = np.outer(sin_x_vals, sin_y_vals)
        grid_2D = np.array(np.meshgrid(x_grid, y_grid)).T

        # try 1st and 1st derivs
        test_11 = False
        if test_11:
            ref_vals = np.outer(cos_x_vals, cos_y_vals)
            for ord in range(3, 7, 2):
                n = (1, 1)
                vals = finite_difference(grid_2D, vals_2D, n, stencil = ord, end_point_accuracy=1)

                errs = self.get_error(ref_vals, vals)
                if plot_error:
                    self.plot_err(grid_2D, ref_vals, vals, errs)
                if print_error:
                    self.print_error(n, ord, errs)
                self.assertLess(errs[1], .1 / ord)

        # 1,2 mixed deriv
        test_12 = False
        if test_12:
            ref_vals = -np.outer(cos_x_vals, sin_y_vals)
            for ord in range(3, 7, 2):
                n = (1, 2)
                vals = finite_difference(grid_2D, vals_2D, n, stencil=ord, end_point_accuracy=1)

                errs = self.get_error(ref_vals, vals)
                if plot_error:
                    self.plot_err(grid_2D, ref_vals, vals, errs)
                if print_error:
                    self.print_error(n, ord, errs)
                self.assertLess(errs[1], .05 / ord)

        # 2,2 mixed deriv
        test_23 = True
        if test_23:
            only_core = True
            ref_vals = np.outer(sin_x_vals, cos_y_vals)
            for ord in range(5, 10):
                n = (2, 3)
                vals = finite_difference(grid_2D, vals_2D, n,
                                         stencil=ord, end_point_accuracy=2,
                                         only_core = True
                                         )

                floop = np.math.floor(ord/2)
                if only_core:
                    refs = ref_vals[floop:-floop, floop:-floop]
                    grfs = grid_2D[floop:-floop, floop:-floop]
                else:
                    refs = ref_vals
                    grfs = grid_2D

                errs = self.get_error(refs, vals)
                if plot_error:
                    self.plot_err(grfs, refs, vals, errs)
                if print_error:
                    self.print_error(n, ord, errs)
                self.assertLess(errs[1], .05)


        # 1,4 mixed deriv
        test_14 = True
        if test_14:
            ref_vals = np.outer(cos_x_vals, sin_y_vals)
            only_core = True
            for ord in range(6, 8):
                n = (1, 4)
                # vals = finite_difference(grid_2D, vals_2D, n,
                #                          stencil=ord, end_point_accuracy=2,
                #                          only_core = only_core
                #                          )
                vals = finite_difference(grid_2D, vals_2D, n,
                                         stencil=ord, end_point_accuracy=2,
                                         only_core = only_core
                                         )


                floop = np.math.floor(ord / 2)
                if only_core:
                    refs = ref_vals[floop:-floop, floop:-floop]
                    grfs = grid_2D[floop:-floop, floop:-floop]
                else:
                    refs = ref_vals
                    grfs = grid_2D

                errs = self.get_error(refs, vals)
                if plot_error:
                    self.plot_err(grfs, refs, vals, errs)
                if print_error:
                    self.print_error(n, ord, errs)
                self.assertLess(errs[1], .5)

    @validationTest
    def test_FD2D_multi(self):

        print_error = False
        plot_error = False

        x_grid = np.arange(0, 1, .01, dtype=np.longdouble)
        y_grid = np.arange(0, 1, .001, dtype=np.longdouble)
        sin_x_vals = np.sin(x_grid); sin_y_vals =  np.sin(y_grid)
        cos_x_vals = np.cos(x_grid); cos_y_vals =  np.cos(y_grid)
        vals_2D = np.outer(sin_x_vals, sin_y_vals)
        grid_2D = np.array(np.meshgrid(x_grid, y_grid)).T

        # try 1st and 1st derivs
        test_11 = True
        if test_11:
            ref_vals = np.outer(cos_x_vals, cos_y_vals)
            for ord in range(3, 7, 2):
                n = (1, 1)
                nreps = 15
                vals = finite_difference(
                    np.broadcast_to(grid_2D, (nreps,) + grid_2D.shape),
                    np.broadcast_to(vals_2D, (nreps,) + vals_2D.shape),
                    n,
                    stencil=ord,
                    end_point_accuracy=1,
                    contract = True
                )

                errs = self.get_error(ref_vals, vals)
                if plot_error:
                    self.plot_err(grid_2D, ref_vals, vals, errs)
                if print_error:
                    self.print_error(n, ord, errs)
                self.assertLess(errs[1], .1 / ord)

    @validationTest
    def test_FDDeriv(self):
        # ggrid = GraphicsGrid(nrows=1, ncols=2)

        fdf = FiniteDifferenceDerivative(np.sin, stencil = 5)
        grid = np.linspace(0, 2*np.pi, 5)
        diff_fun = fdf(grid, mesh_spacing = .1)
        d = np.array([-.2, -.1, 0, .1, .2])
        raw = np.array([
            finite_difference(
                c + d,
                np.sin(c + d),
                (1,),
                stencil=5, only_center = True, end_point_accuracy=0
                ) for c in grid
        ])

        self.assertLess(np.linalg.norm(diff_fun[0] - raw), .00001)
        self.assertIsInstance(diff_fun[0, 0, 0][0], (float,))

    @validationTest
    def test_FDDeriv2(self):
        def test(gps):
            x = gps[..., 0]
            y = gps[..., 1]

            return x * np.sin(y)

        fdf = FiniteDifferenceDerivative(test, function_shape=((2, ), 1), stencil = 5)

        npts = 25
        grid = np.linspace(0, 2*np.pi, npts)
        grid2D = np.meshgrid(grid, grid)
        gps = np.array(grid2D).T
        diff_fun = fdf(gps)

        diff_fun[0, 0]

        self.assertEquals(diff_fun[0, 0].shape, (npts, npts))

        self.assertEquals(diff_fun[0, :].shape, (1, 2, npts, npts)) # should I implement this with a squeeze...?
        self.assertEquals(np.linalg.norm((diff_fun[:, :][1, 0] - diff_fun[1, 0]).flatten()), 0.)

        # gg =  GraphicsGrid(nrows=2, ncols=4, graphics_class=Graphics3D, fig_kw={'figsize':(15, 8)})

        # for i in range(2):
        #     for j in range(2):
        #         Plot3D(
        #             *grid2D,
        #             np.round(diff_fun[i, j].reshape(npts, npts), 5),
        #             plot_style={
        #                 "cmap" : "viridis"
        #             },
        #             axes_labels = ['x', 'y', 'z'],
        #             figure= gg[i, j]
        #         )
        #         num_x = (1 if i == 0 else 0) + (1 if j == 0 else 0) # can be done with mod but too tired to thing
        #         num_y = (0 if i == 0 else 1) + (0 if j == 0 else 1) # can be done with mod but too tired to thing
        #         Plot3D(
        #             *grid2D,
        #             #test(gps),
        #             np.round(finite_difference(gps, test(gps), (num_x, num_y), (5, 5)), 5),
        #             plot_style={
        #                 "cmap" : "viridis"
        #             },
        #             axes_labels = ['x', 'y', 'z'],
        #             figure=gg[i, 2+j]
        #         )
        # gg.show()

    #endregion

    #region Mesh

    @validationTest
    def test_LinSpaceMesh(self):
        mg = Mesh(np.linspace(-1, 1, 3))
        self.assertIs(mg.mesh_type, MeshType.Structured)
        self.assertEquals(mg.shape, (3,))

    @validationTest
    def test_MeshGridMesh(self):
        mg = np.meshgrid(np.array([-1, 0, 1]), np.array([-1, 0, 1]))
        regmesh = Mesh(mg)
        self.assertIs(regmesh.mesh_type, MeshType.Structured)
        self.assertEquals(regmesh.shape, (3, 3, 2))

    @validationTest
    def test_RegularMesh(self):
        regmesh = Mesh.RegularMesh([-1, 1, 3], [-1, 1, 3])
        self.assertIs(regmesh.mesh_type, MeshType.Structured)
        self.assertEquals(regmesh.shape, (3, 3, 2))

    @validationTest
    def test_RegMeshSubgrids(self):
        regmesh = Mesh.RegularMesh([-1, 1, 3], [-1, 1, 3])
        m = [Mesh(g) for g in regmesh.subgrids]
        self.assertTrue(all(g.mesh_type is MeshType.Structured for g in m))

    @inactiveTest
    def test_SemiStructuredMesh(self):
        # class MeshType(enum.Enum):
        #     Structured = "structured"
        #     Unstructured = "unstructured"
        #     SemiStructured = "semistructured"
        #     Indeterminate = "indeterminate"
        semi_mesh = Mesh([[np.arange(i)] for i in range(10, 25)])
        self.assertIs(semi_mesh.mesh_type, MeshType.SemiStructured)

    @validationTest
    def test_MeshFromList(self):
        try:
            wat = np.asarray([[-1, 0, 1], [-1, 0, 1]])
            bad = Mesh(wat)
        except Mesh.MeshError:
            pass
        else:
            # just to make it accessible through Mesh
            self.assertIs(bad.mesh_type, MeshType.Indeterminate)

    #endregion Mesh

    #region Tensor Derivatives
    @validationTest
    def test_TensorTerms(self):

        n_Q = 10
        n_X = 20
        x_derivs = [
            np.random.rand(n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_Q, n_X)
            ]
        v_derivs = [
            np.random.rand(n_X),
            np.random.rand(n_X, n_X),
            np.random.rand(n_X, n_X, n_X),
            np.random.rand(n_X, n_X, n_X, n_X)
            ]
        terms = TensorExpansionTerms(x_derivs, v_derivs)

        q1 = terms.QX(1)
        self.assertEquals(str(q1), 'Q[1]')

        q11 = terms.QX(1) + terms.QX(1)
        self.assertEquals(str(q11), 'Q[1]+Q[1]')

        qshift = q11.shift(2, 1)
        self.assertEquals(str(qshift), '[Q[1]+Q[1]|2->1]')

        vQ = terms.QX(1).dot(terms.XV(1), 2, 1)
        self.assertEquals(str(vQ), '<Q[1]:2,1:V[1]>')

        self.assertIsInstance(vQ, TensorExpansionTerms.ContractionTerm)
        # self.assertIsInstance(vQ.simplify(), TensorExpansionTerms.BasicContractionTerm)

        vdQQ = vQ.dQ()
        self.assertIn("V[2]", str(vdQQ))
        self.assertIn("<Q[2]:3,1:V[1]>", str(vdQQ))

        vdQQQQ = vdQQ.dQ().dQ().reduce_terms()
        self.assertIsInstance(vdQQQQ, TensorExpansionTerms.SumTerm)
        self.assertEquals(len(vdQQQQ.terms), 15)

        # arr = vdQQQQ.array

        xv_derivs = [
            [None, np.random.rand(n_Q, n_X, n_X)],
            [None, np.random.rand(n_Q, n_Q, n_X, n_X)]
            ]

        mixed_terms = TensorExpansionTerms(x_derivs, v_derivs, qxv_terms=xv_derivs)

        vdQ = mixed_terms.QX(1).dot(mixed_terms.XV(1), 2, 1)

        mixed_vdQXX = vdQ.dQ().dQ().reduce_terms()

        vdQQQQ_subbed = mixed_vdQXX.dQ().reduce_terms()

        self.assertEquals(len(vdQQQQ_subbed.terms), 14)

        self.assertEquals(vdQQQQ_subbed.array.shape, (n_Q, n_Q, n_Q, n_Q))

    def test_TensorConversion(self):

        n_Q = 10
        n_X = 20
        np.random.seed(1)
        x_derivs = [
            np.random.rand(n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_Q, n_X)
        ]
        v_derivs = [
            np.random.rand(n_X),
            np.random.rand(n_X, n_X),
            np.random.rand(n_X, n_X, n_X),
            np.random.rand(n_X, n_X, n_X, n_X)
        ]
        xv_derivs = [
            [None, np.random.rand(n_Q, n_X, n_X)],
            [None, np.random.rand(n_Q, n_Q, n_X, n_X)]
            ]

        t = TensorDerivativeConverter(x_derivs, v_derivs)
        new = t.convert()
        self.assertEquals(len(new), 4)
        self.assertEquals(new[1].shape, (n_Q, n_Q))

        t2 = TensorDerivativeConverter(x_derivs, v_derivs, mixed_terms=xv_derivs)
        new = t2.convert()
        self.assertEquals(len(new), 4)
        self.assertEquals(new[1].shape, (n_Q, n_Q))


        trace_derivs = TensorExpansionTerms([0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
        t1 = trace_derivs.QX(1).tr().dQ()
        self.assertEquals(str(t1), 'Tr[Q[2],2+3]')

        t1 = trace_derivs.QX(1).det().dQ().dQ()

        t1 = trace_derivs.QX(1).inverse().dQ()

        # raise Exception(t1)
        # raise Exception(t1)


    @debugTest
    def test_PseudopotentialTerms(self):

        n_Q = 10
        np.random.seed(1)
        n_X = 3

        i_derivs = [
            np.random.rand(n_X, n_X),
            np.random.rand(n_Q, n_X, n_X),
            np.random.rand(n_Q, n_Q, n_X, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_X, n_X)
        ]
        # for x in i_derivs: # symmetrize
        #     inds = np.indices(x.shape).T.reshape(-1, x.ndim)
        #     for i in inds:
        #         n_X_part = i[-2:]
        #         n_Q_part = i[:-2]
        #         parent = np.concatenate([np.sort(n_Q_part), np.sort(n_X_part)])
        #         x[i] = x[parent]
        i_terms = TensorExpansionTerms(i_derivs[1:], None, base_qx=i_derivs[0], q_name='I')
        detI = i_terms.QX(0).det()

        g_derivs = [
            np.random.rand(n_Q, n_Q),
            np.random.rand(n_Q, n_Q, n_Q),
            np.random.rand(n_Q, n_Q, n_Q, n_Q),
            np.random.rand(n_Q, n_Q, n_Q, n_Q, n_Q)
        ]
        # for x in g_derivs: # symmetrize
        #     inds = np.indices(x.shape).T.reshape(-1, x.ndim)
        #     for i in inds:
        #         n_X_part = i[-2:]
        #         n_Q_part = i[:-2]
        #         parent = np.concatenate([np.sort(n_Q_part), np.sort(n_X_part)])
        #         x[i] = x[parent]
        g_terms = TensorExpansionTerms(g_derivs[1:], None, base_qx=g_derivs[0], q_name='G')
        detG = g_terms.QX(0).det()
        self.assertIsInstance(detG.array, float)
        self.assertEquals(detG.array, np.linalg.det(g_derivs[0]))

        new_tr = g_terms.QX(0).tr()
        self.assertEquals(new_tr.array.shape, ())
        self.assertTrue(np.allclose(new_tr.array, np.trace(g_derivs[0])))

        new_inv = ~g_terms.QX(0)
        self.assertTrue(np.allclose(new_inv.array, np.linalg.inv(g_derivs[0])))

        newdQ = detG.dQ()
        self.assertEquals(newdQ.array.shape, (n_Q,))

        i_derivs = [
            np.random.rand(n_X, n_X),
            np.random.rand(n_Q, n_X, n_X),
            np.random.rand(n_Q, n_Q, n_X, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_X, n_X)
        ]
        # for x in i_derivs: # symmetrize
        #     inds = np.indices(x.shape).T.reshape(-1, x.ndim)
        #     for i in inds:
        #         n_X_part = i[-2:]
        #         n_Q_part = i[:-2]
        #         parent = np.concatenate([np.sort(n_Q_part), np.sort(n_X_part)])
        #         x[i] = x[parent]
        i_terms = TensorExpansionTerms(i_derivs[1:], None, base_qx=i_derivs[0], q_name='I')
        detI = i_terms.QX(0).det()

        gam = detG / detI
        self.assertEquals(gam.array.shape, ())
        self.assertTrue(np.allclose(gam.array, detG.array / detI.array))
        self.assertTrue(np.allclose(gam.array, np.linalg.det(g_derivs[0]) / np.linalg.det(i_derivs[0])))


        five_gam = 5 * detG / detI
        self.assertAlmostEquals(five_gam.array, 5 * detG.array / detI.array, 8)

        inv_gam = 1/gam
        self.assertEquals(inv_gam.array.shape, ())
        self.assertTrue(np.allclose(inv_gam.array, detI.array / detG.array))
        self.assertEquals(inv_gam.array, 1/gam.array)

        gamdQ = gam.dQ().simplify(check_arrays=True)

        self.assertEquals(gamdQ.array.shape, (n_Q,))
        # self.assertEquals(gamdQQ.array.shape, (n_Q, n_Q))

        # wat = g_terms.QX(0).dot(gamdQQ, [1, 2], [1, 2])
        # self.assertEquals(wat.array.shape, ())

        wat_2 = g_terms.QX(1).dot(gamdQ, 3, 1).tr()
        self.assertEquals(wat_2.array.shape, ())
        wat_21 = g_terms.QX(2).tr(axis1=1, axis2=4)
        self.assertEquals(wat_21.array.shape, (n_Q, n_Q))
        self.assertTrue(np.allclose(wat_21.array, np.trace(g_derivs[2], axis1=0, axis2=3)))


        doot = gamdQ.dot(g_terms.QX(0), 1, 1)
        self.assertEquals(doot.array.shape, (n_Q,))

        wat_3 = -3 / 4 * gamdQ.dot(doot, 1, 1)
        self.assertEquals(wat_3.array.shape, ())

        wat_4 = -3 / 4 * inv_gam * gamdQ.dot(doot, 1, 1)
        self.assertEquals(wat_4.array.shape, ())

        wat_5 = inv_gam * wat_3
        self.assertTrue(np.allclose(wat_4.array, wat_5.array))

        I0, I0Q, I0QQ = i_derivs[:3]
        G, GQ, GQQ = g_derivs[:3]

        detI = np.linalg.det(I0); invI = np.linalg.inv(I0); adjI = invI * detI
        detG = np.linalg.det(G); invG = np.linalg.inv(G); adjG = invG * detG


        invIdQ_new = i_terms.QX(0).inverse().dQ()
        invIdQ = - np.tensordot(np.tensordot(invI, I0Q, axes=[-1, 1]), invI, axes=[-1, 0]).transpose(1, 0, 2)
        self.assertEquals(invIdQ_new.array.shape, invIdQ.shape)
        self.assertTrue(np.allclose(invIdQ_new.array, invIdQ))

        invGdQ = - np.tensordot(np.tensordot(invG, GQ, axes=[-1, 1]), invG, axes=[-1, 0]).transpose(1, 0, 2)

        # not quite enough terms to want to be clever here...
        nQ = GQ.shape[0]
        ## First derivatives of the determinant
        detIdQ = np.array([
            np.trace(np.dot(adjI, I0Q[i]))
            for i in range(nQ)
        ])
        detIdQ2 = np.trace(
            np.tensordot(adjI, I0Q, axes=[1, 1]),
            axis1=0,
            axis2=2
        )
        detI_new = i_terms.QX(0).det()
        detIdQ_new = detI_new.dQ()
        self.assertTrue(np.allclose(detIdQ2, detIdQ))
        self.assertEquals(detIdQ_new.array.shape, detIdQ.shape)
        self.assertTrue(np.allclose(detIdQ_new.array, detIdQ))

        detGdQ = np.array([
            np.trace(np.dot(adjG, GQ[i]))
            for i in range(nQ)
        ])
        detG_new = g_terms.QX(0).det()
        detGdQ_new = detG_new.dQ()
        self.assertEquals(detGdQ_new.array.shape, detGdQ.shape)
        self.assertTrue(np.allclose(detGdQ_new.array, detGdQ))

        # two_dot = g_terms.QX(0).dot(gamdQQ, [1, 2], [1, 2])
        # self.assertTrue(np.allclose(two_dot.array, np.tensordot(g_terms.QX(0).array, gamdQQ.array, axes=[[0, 1], [0, 1]])))
        gamdQ = (detI_new.dQ()/detI_new + -1 * detG_new.dQ()/detG_new).simplify(check_arrays=True)

        # gamdQQ = gamdQ.dQ().simplify(check_arrays=True)

        ## Derivatives of ln Gamma
        gamdQ_I = 1 / detI * detIdQ
        gamdQ_G = 1 / detG * detGdQ
        gamdQ_og = gamdQ_I - gamdQ_G

        self.assertTrue(np.allclose(gamdQ.array, gamdQ_og))

        adjIdQ = detI * invIdQ + detIdQ[:, np.newaxis, np.newaxis] * invI[np.newaxis, :, :]
        np.array([
            np.trace(np.dot(adjI, I0Q[i]))
            for i in range(nQ)
        ])
        detIdQQ = np.array([
            [
                np.tensordot(I0Q[i], adjIdQ[j], axes=2)
                + np.tensordot(adjI, I0QQ[i, j], axes=2)
                for i in range(nQ)
            ]
            for j in range(nQ)
        ])

        detIdQQ2_terms = [
            np.array([
                [
                    np.tensordot(I0Q[i], adjIdQ[j], axes=2)
                    for i in range(nQ)
                ]
                for j in range(nQ)
            ]),
            np.array([
                [
                    np.tensordot(adjI, I0QQ[i, j], axes=2)
                    for i in range(nQ)
                ]
                for j in range(nQ)
            ])
        ]
        detIdQQ2 = np.sum(detIdQQ2_terms, axis=0)

        detIdQQ_new = detIdQ_new.dQ().simplify(check_arrays=True)

        adjI_new = i_terms.QX(0).det() * i_terms.QX(0).inverse()
        self.assertTrue(np.allclose(adjI_new.array, adjI))

        adjIdQ_new = adjI_new.dQ().simplify()
        # raise Exception(adjIdQ_new.terms[0])
        self.assertTrue(np.allclose(adjIdQ_new.array, adjIdQ))

        # raise Exception(detIdQQ_new, detIdQQ2_terms[1])
        self.assertTrue(np.allclose(detIdQQ2, detIdQQ))
        # raise Exception(detIdQQ_new)
        # raise Exception
        raise Exception(detIdQQ_new.array, detIdQQ)
        self.assertEquals(detIdQQ_new.array.shape, detIdQQ.shape)
        self.assertTrue(np.allclose(detIdQQ_new.array, detIdQQ))

        gamdQQ_I = -1 / detI ** 2 * np.outer(detIdQ, detIdQ) + 1 / detI * detIdQQ

    #endregion Tensor Derivatives

    #region Function expansions

    @validationTest
    def test_ExpandFunction(self):
        dtype = np.float32

        def sin_xy(pt):
            ax = -1 if pt.ndim > 1 else 0
            return np.prod(np.sin(pt), axis=ax)

        point = np.array([.5, .5], dtype=dtype)
        exp = FunctionExpansion.expand_function(sin_xy, point, function_shape=((2,), 0), order=4, stencil=6)

        hmm = np.vstack(
            [np.linspace(-.5, .5, 100, dtype=dtype), np.zeros((100,), dtype=dtype)]
        ).T + point[np.newaxis]
        # print(hmm)
        ref = sin_xy(hmm)
        test = exp(hmm)

        plot_error = False
        if plot_error:
            exp2 = FunctionExpansion.expand_function(sin_xy, point, function_shape=((2,), 0), order=1, stencil=5)
            g = hmm[:, 0]
            gg = GraphicsGrid(nrows=1, ncols=2, tighten=True)
            gg[0, 0] = Plot(g, exp(hmm), figure=Plot(g, sin_xy(hmm), figure=gg[0, 0]))
            gg[0, 1] = Plot(g, exp2(hmm), figure=Plot(g, sin_xy(hmm), figure=gg[0, 1]))
            gg.show()

        plot2Derror = False
        if plot2Derror:
            # -0.008557147793556821113 0.007548466137326598213 0.0019501675856203966399
            mesh = np.meshgrid(
                np.linspace(.4, .6, 100, dtype=dtype),
                np.linspace(.4, .6, 100, dtype=dtype)
            )
            grid = np.array(mesh).T
            gg = GraphicsGrid(nrows=1, ncols=2, tighten=True)
            ref2 = sin_xy(grid)
            test2 = exp(grid)
            err2 = ref2 - test2
            # print(np.min(err2), np.max(err2), np.average(np.abs(err2)))
            gg[0, 1] = ContourPlot(*mesh, ref2 - test2, figure=gg[0, 1], plot_style=dict(vmin=-.2, vmax=.2))
            gg[0, 0] = ContourPlot(*mesh, test2, figure=gg[0, 0])
            gg.show()

        self.assertEquals(exp(point), exp.ref)
        self.assertLess(np.linalg.norm(test - ref), .01)

    #endregion

