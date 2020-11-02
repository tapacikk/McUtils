
from Peeves.TestUtils import *
from McUtils.Numputils import *
from McUtils.Zachary import FiniteDifferenceDerivative
from unittest import TestCase
import numpy as np

class NumputilsTests(TestCase):

    @validationTest
    def test_SparseArray(self):
        array = SparseArray([
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ],
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ],
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 1]
            ],
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]
            ]
        ])

        self.assertEquals(array.shape, (4, 3, 3))
        tp = array.transpose((1, 0, 2))
        self.assertEquals(tp.shape, (3, 4, 3))
        self.assertLess(np.linalg.norm((tp.todense()-array.todense().transpose((1, 0, 2))).flatten()), 1e-8)
        self.assertEquals(array[2, :, 2].shape, (3,))
        td = array.tensordot(array, axes=[1, 1])
        self.assertEquals(td.shape, (4, 3, 4, 3))
        self.assertEquals(array.tensordot(array, axes=[[1, 2], [1, 2]]).shape, (4, 4))

    @validationTest
    def test_PtsAngleDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        ang, deriv, deriv_2 = angle_deriv(coords, 5, 4, 6, order=2)
        ang2 = vec_angles(coords[4] - coords[5], coords[6] - coords[5])[0]

        self.assertEquals(ang2, ang.flatten()[0])

        fd = FiniteDifferenceDerivative(
            lambda pt: vec_angles(pt[..., 1, :] - pt[..., 0, :], pt[..., 2, :] - pt[..., 0, :])[0],
            function_shape=((None, 3), 0),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(5, 4, 6),]).derivative_tensor([1, 2])

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        deriv_2 = deriv_2.squeeze()

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1], deriv_2[0, 2]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1], deriv_2[1, 2]], axis=1),
                np.concatenate([deriv_2[2, 0], deriv_2[2, 1], deriv_2[2, 2]], axis=1)
            ],
            axis=0
        )

        # raise Exception(deriv_2[1, 0], deriv_2[0, 1])
        # import McUtils.Plots as plt

        # plt.ArrayPlot(d2_flat).show()

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat, fd2)))

        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))

        # raise Exception(fd2.flatten(), deriv_2.flatten())

    @debugTest
    def test_NormDerivs(self):
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        a = coords[(4, 5),] - coords[(5, 4),]
        na, na_da, na_daa = vec_norm_derivs(a, order=2)
        na_2 = vec_norms(a)

        self.assertEquals(tuple(na_2), tuple(na))


        norm_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_norms(vecs),
            function_shape=((None, 2, 3), (None,)),
            mesh_spacing=1.0e-4
        )

        fd_nada, fd_nadaa = norm_fd(a).derivative_tensor([1, 2])

        fd_nada = np.array([fd_nada[:3, 0], fd_nada[3:, 1]])
        fd_nadaa = np.array([fd_nadaa[:3, :3, 0], fd_nadaa[3:, 3:, 1]])

        self.assertTrue(np.allclose(na_da.flatten(), fd_nada.flatten()), msg="norm d1: {} and {} differ".format(
            na_da.flatten(), fd_nada.flatten()
        ))

        self.assertTrue(np.allclose(na_daa.flatten(), fd_nadaa.flatten(), atol=1.0e-4), msg="norm d1: {} and {} differ".format(
            na_daa.flatten(), fd_nadaa.flatten()
        ))

    @debugTest
    def test_SinCosDerivs(self):
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        # sin_derivs_extra, cos_derivs_extra = vec_sin_cos_derivs( # just here to make sure the shape works
        #     coords[(1, 2, 3, 4),] - coords[(0, 1, 2, 3),],
        #     coords[(2, 3, 4, 5),] - coords[(0, 1, 2, 3),],
        #     order=2)

        a = coords[6] - coords[5]
        b = coords[4] - coords[5]

        sin_derivs, cos_derivs = vec_sin_cos_derivs(np.array([a, b]), np.array([b, a]), order=2)

        cos_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_cos(vecs[..., 0, :], vecs[..., 1, :]),
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-7
        )
        cosDeriv_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_sin_cos_derivs(vecs[..., 0, :], vecs[..., 1, :], order=1)[1][1].squeeze(),
            function_shape=((None, 2, 3), (None, 2, 3)),
            mesh_spacing=1.0e-7
        )
        cos_fd22_1, = cosDeriv_fd(np.array([a, b])).derivative_tensor([1])
        cos_fd22_2, = cosDeriv_fd(np.array([b, a])).derivative_tensor([1])
        cos_fd22 = np.array([cos_fd22_1, cos_fd22_2])

        cos_fd1_1, cos_fd2_1 = cos_fd(np.array([a, b])).derivative_tensor([1, 2])
        cos_fd1_2, cos_fd2_2 = cos_fd(np.array([b, a])).derivative_tensor([1, 2])
        cos_fd1 = np.array([cos_fd1_1, cos_fd1_2])
        cos_fd2 = np.array([cos_fd2_1, cos_fd2_2])

        sin_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_sins(vecs[..., 0, :], vecs[..., 1, :]),
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-7
        )
        sinDeriv_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_sin_cos_derivs(vecs[..., 0, :], vecs[..., 1, :], order=1)[0][1].squeeze(),
            function_shape=((None, 2, 3), (None, 2, 3)),
            mesh_spacing=1.0e-7
        )
        sin_fd22_1, = sinDeriv_fd(np.array([a, b])).derivative_tensor([1])
        sin_fd22_2, = sinDeriv_fd(np.array([b, a])).derivative_tensor([1])
        sin_fd22 = np.array([sin_fd22_1, sin_fd22_2])
        sin_fd1_1, sin_fd2_1 = sin_fd(np.array([a, b])).derivative_tensor([1, 2])
        sin_fd1_2, sin_fd2_2 = sin_fd(np.array([b, a])).derivative_tensor([1, 2])
        sin_fd1 = np.array([sin_fd1_1, sin_fd1_2])
        sin_fd2 = np.array([sin_fd2_1, sin_fd2_2])

        s, s1, s2 = sin_derivs
        c, c1, c2 = cos_derivs

        # raise Exception(sin_fd1, s1)

        self.assertTrue(np.allclose(s1.flatten(), sin_fd1.flatten()), msg="sin d1: {} and {} differ".format(
            s1.flatten(), sin_fd1.flatten()
        ))

        self.assertTrue(np.allclose(c1.flatten(), cos_fd1.flatten()), msg="cos d1: {} and {} differ".format(
            c1.flatten(), cos_fd1.flatten()
        ))

        # raise Exception("\n", c2[0, 0], "\n", cos_fd2[:3, :3])
        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(c2[0, 0], cos_fd2[:3, :3])))
        c2_flat = np.concatenate(
            [
                np.concatenate([c2[:, 0, 0], c2[:, 0, 1]], axis=2),
                np.concatenate([c2[:, 1, 0], c2[:, 1, 1]], axis=2)
            ],
            axis=1
        )

        s2_flat = np.concatenate(
            [
                np.concatenate([s2[:, 0, 0], s2[:, 0, 1]], axis=2),
                np.concatenate([s2[:, 1, 0], s2[:, 1, 1]], axis=2)
            ],
            axis=1
        )

        # print("???????")

        # test_sin, test_cos = vec_sin_cos_derivs(a, b, order=2)
        #
        # tc, tc1, tc2 = test_cos
        # ts, ts1, ts2 = test_sin
        #
        # tc2_flat = np.concatenate(
        #     [
        #         np.concatenate([tc2[0, 0], tc2[0, 1]], axis=1),
        #         np.concatenate([tc2[1, 0], tc2[1, 1]], axis=1)
        #     ],
        #     axis=0
        # )

        # raise Exception(c2_flat[0] - tc2_flat)

        # raise Exception("\n"+"\n".join("{} {} {}".format(a, b, c) for a, b, c in zip(s2_flat, sin_fd22.reshape((6, 6)), sin_fd2)))

        # raise Exception("\n"+"\n".join("{} {} {}".format(a, b, c) for a, b, c in zip(c2_flat, cos_fd2, cos_fd22.reshape((6, 6)))))
        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(c2[1, 1], cos_fd2[3:, 3:])))
        # raise Exception(c2[1, 0], c2[0, 1].T, cos_fd2[3:, :3])

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(s2_flat, sin_fd2)))

        # raise Exception(s2[0, 0], sin_fd2[:3, :3])
        # raise Exception(s2[1, 0], s2[0, 1].T, sin_fd2[3:, :3], sin_fd2[:3, 3:].T)
        # raise Exception(s2[1, 1], sin_fd2[3:, 3:])
        # raise Exception(s2[0, 1], sin_fd2[:3, 3:])
        self.assertTrue(np.allclose(s2_flat.flatten(), sin_fd22.flatten()), msg="sin d2: {} and {} differ".format(
            s2_flat.flatten(), sin_fd22.flatten()
        ))
        self.assertTrue(np.allclose(c2_flat.flatten(), cos_fd22.flatten()), msg="cos d2: {} and {} differ".format(
            c2_flat.flatten(), cos_fd22.flatten()
        ))

    @validationTest
    def test_AngleDerivs(self):
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        a = coords[4] - coords[5]
        b = coords[6] - coords[5]
        ang, dang, ddang = vec_angle_derivs(a, b, order=2)
        ang_2 = vec_angles(a, b)[0]

        self.assertEquals(ang_2, ang.flatten()[0])

        ang_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_angles(vecs[..., 0, :], vecs[..., 1, :])[0],
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-4
        )

        fd_ang, fd_dang = ang_fd([a, b]).derivative_tensor([1, 2])

        self.assertTrue(np.allclose(dang.flatten(), fd_ang.flatten()), msg="ang d1: {} and {} differ".format(
            fd_ang.flatten(), fd_ang.flatten()
        ))


        d2_flat = np.concatenate(
            [
                np.concatenate([ddang[0, 0], ddang[0, 1]], axis=1),
                np.concatenate([ddang[1, 0], ddang[1, 1]], axis=1)
            ],
            axis=0
        )

        self.assertTrue(np.allclose(d2_flat.flatten(), fd_dang.flatten(), atol=1.0e-3), msg="cos d2: {} and {} differ".format(
            d2_flat.flatten(), fd_dang.flatten()
        ))

