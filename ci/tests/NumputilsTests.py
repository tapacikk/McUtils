
from Peeves.TestUtils import *
from Peeves import BlockProfiler
from McUtils.Numputils import *
from McUtils.Zachary import FiniteDifferenceDerivative
from unittest import TestCase
import numpy as np

class NumputilsTests(TestCase):

    problem_coords = np.array([
                                  [-1.86403557e-17, -7.60465240e-02,  4.62443228e-02],
                                  [ 6.70904773e-17, -7.60465240e-02, -9.53755677e-01],
                                  [ 9.29682337e-01,  2.92315732e-01,  4.62443228e-02],
                                  [ 2.46519033e-32, -1.38777878e-17,  2.25076602e-01],
                                  [-1.97215226e-31,  1.43714410e+00, -9.00306410e-01],
                                  [-1.75999392e-16, -1.43714410e+00, -9.00306410e-01]
    ])
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
    def test_ProblemPtsAllDerivs(self):
        import McUtils.Numputils.Options as NumOpts

        NumOpts.zero_placeholder = np.inf

        coords = self.problem_coords

        # dists, dist_derivs, dist_derivs_2 = dist_deriv(coords, [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], order=2)
        # angs, ang_derivs, ang_derivs_2 = angle_deriv(coords,
        #                                              [0, 1, 2, 3, 4, 5],
        #                                              [1, 2, 3, 4, 5, 0],
        #                                              [2, 3, 4, 5, 0, 1],
        #                                              order=2
        #                                              )
        # diheds, dihed_derivs, dihed_derivs_2 = dihed_deriv(coords,
        #                                                [0, 1, 2, 3, 4, 5],
        #                                                [1, 2, 3, 4, 5, 0],
        #                                                [2, 3, 4, 5, 0, 1],
        #                                                [3, 4, 5, 0, 1, 2],
        #                                                order=2
        #                                                )

        diheds, dihed_derivs, dihed_derivs_2 = dihed_deriv(coords,
                                                           [3],
                                                           [4],
                                                           [5],
                                                           [0],
                                                           order=2
                                                           )

        # raise Exception([
        #     diheds,
        #     [np.min(dihed_derivs), np.max(dihed_derivs)],
        #     [np.min(dihed_derivs_2), np.max(dihed_derivs_2)]
        #     ])
        #
        # raise Exception(
        #     np.array([dists.flatten(), np.round(np.rad2deg(angs.flatten()), 1), np.round(np.rad2deg(diheds.flatten()), 1)]),
        #     [
        #         [np.min(dist_derivs), np.max(dist_derivs)],
        #         [np.min(dist_derivs_2), np.max(dist_derivs_2)]
        #         ],
        #     [
        #         [np.min(ang_derivs), np.max(ang_derivs)],
        #         [np.min(ang_derivs_2), np.max(ang_derivs_2)]
        #         ],
        #     [
        #         [np.min(dihed_derivs), np.max(dihed_derivs)],
        #         [np.min(dihed_derivs_2), np.max(dihed_derivs_2)]
        #     ]
        # )

    @validationTest
    def test_PtsDihedralsDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        angs, derivs, derivs_2 = dihed_deriv(coords, [4, 7], [5, 6], [6, 5], [7, 4], order=2)
        ang = angs[0]; deriv = derivs[:, 0, :]; deriv_2 = derivs_2[:, :, 0, :, :]
        ang2 = pts_dihedrals(coords[4],  coords[5], coords[6], coords[7])

        self.assertEquals(ang2, ang[0])

        fd = FiniteDifferenceDerivative(
            lambda pt: pts_dihedrals(pt[..., 0, :], pt[..., 1, :], pt[..., 2, :], pt[..., 3, :]),
            function_shape=((None, 4, 3), 0),
            mesh_spacing=1.0e-5
        )
        dihedDeriv_fd = FiniteDifferenceDerivative(
            lambda pts: dihed_deriv(pts, 0, 1, 2, 3, order=1)[1].squeeze().transpose((1, 0, 2)),
            function_shape=((None, 4, 3), (None, 4, 3)),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(4, 5, 6, 7),]).derivative_tensor([1, 2])
        fd2_22 = dihedDeriv_fd(coords[(4, 5, 6, 7),]).derivative_tensor(1)

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1], deriv_2[0, 2], deriv_2[0, 3]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1], deriv_2[1, 2], deriv_2[1, 3]], axis=1),
                np.concatenate([deriv_2[2, 0], deriv_2[2, 1], deriv_2[2, 2], deriv_2[2, 3]], axis=1),
                np.concatenate([deriv_2[3, 0], deriv_2[3, 1], deriv_2[3, 2], deriv_2[3, 3]], axis=1)
            ],
            axis=0
        )

        bleh = fd2_22.reshape(12, 12)
        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(
        #     np.round(deriv_2[2, 2], 3), np.round(bleh[6:9, 6:9], 3))
        #                                ))
        # raise Exception(np.round(d2_flat-bleh, 3))
        # raise Exception("\n"+"\n".join("{}\n{}".format(a, b) for a, b in zip(np.round(d2_flat, 3), np.round(bleh, 3))))
        self.assertTrue(np.allclose(d2_flat.flatten(), bleh.flatten(), atol=1.0e-7), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), bleh.flatten()
        ))
        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))

        # raise Exception(fd2.flatten(), deriv_2.flatten())

    @validationTest
    def test_PtsAngleDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        angs, derivs, derivs_2 = angle_deriv(coords, [5, 5], [4, 6], [6, 4], order=2)

        ang = angs[0]; deriv = derivs[:, 0, :]; deriv_2 = derivs_2[:, :, 0, :, :]
        ang2 = vec_angles(coords[4] - coords[5], coords[6] - coords[5])[0]

        self.assertEquals(ang2, ang)

        fd = FiniteDifferenceDerivative(
            lambda pt: vec_angles(pt[..., 1, :] - pt[..., 0, :], pt[..., 2, :] - pt[..., 0, :])[0],
            function_shape=((None, 3), 0),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(5, 4, 6),]).derivative_tensor([1, 2])

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1], deriv_2[0, 2]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1], deriv_2[1, 2]], axis=1),
                np.concatenate([deriv_2[2, 0], deriv_2[2, 1], deriv_2[2, 2]], axis=1)
            ],
            axis=0
        )

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat, fd2)))

        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))

        # raise Exception(fd2.flatten(), deriv_2.flatten())

    @validationTest
    def test_PtsDistDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        dists, derivs, derivs_2 = dist_deriv(coords, [5, 4], [4, 5], order=2)

        dist = dists[0]; deriv = derivs[:, 0, :]; deriv_2 = derivs_2[:, :, 0, :, :]
        dists2 = vec_norms(coords[4] - coords[5])

        self.assertEquals(dists2, dist)
        # raise Exception(dist, dists2)

        fd = FiniteDifferenceDerivative(
            lambda pt: vec_norms(pt[..., 1, :] - pt[..., 0, :]),
            function_shape=((None, 3), 0),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(5, 4),]).derivative_tensor([1, 2])

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1]], axis=1)
            ],
            axis=0
        )

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat, fd2)))

        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))

        # raise Exception(fd2.flatten(), deriv_2.flatten())

    @validationTest
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

    @validationTest
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
        ang, dang, ddang = vec_angle_derivs(np.array([a, b]),
                                            np.array([b, a]), order=2)
        ang_2 = vec_angles(a, b)[0]

        self.assertEquals(ang_2, ang.flatten()[0])

        ang_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_angles(vecs[..., 0, :], vecs[..., 1, :])[0],
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-4
        )

        fd_ang_1, fd_dang_1 = ang_fd([a, b]).derivative_tensor([1, 2])
        fd_ang_2, fd_dang_2 = ang_fd([b, a]).derivative_tensor([1, 2])

        fd_ang = np.array([fd_ang_1, fd_ang_2])
        fd_dang = np.array([fd_dang_1, fd_dang_2])

        self.assertTrue(np.allclose(dang.flatten(), fd_ang.flatten()), msg="ang d1: {} and {} differ".format(
            fd_ang.flatten(), fd_ang.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([ddang[:, 0, 0], ddang[:, 0, 1]], axis=2),
                np.concatenate([ddang[:, 1, 0], ddang[:, 1, 1]], axis=2)
            ],
            axis=1
        )

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat[0], fd_dang[0])))
        self.assertTrue(np.allclose(d2_flat.flatten(), fd_dang.flatten(), atol=1.0e-2), msg="ang d2: {} and {} differ ({})".format(
            d2_flat.flatten(), fd_dang.flatten(), d2_flat.flatten() - fd_dang.flatten()
        ))

    @inactiveTest
    def test_AngleDerivScan(self):
        np.random.seed(0)
        # a = np.random.rand(3) * 2 # make it longer to avoid stability issues
        # dump_file = "/Users/Mark/Desktop/wat2.json"
        a = np.array([1, 0, 0]); dump_file = "/Users/Mark/Desktop/wat.json"

        fd = FiniteDifferenceDerivative(
                lambda vecs: vec_angle_derivs(vecs[..., 0, :], vecs[..., 1, :], up_vectors=up)[1],
                function_shape=((None, 2, 3), 0),
                mesh_spacing=1.0e-4
            )

        data = {"rotations":[], 'real_angles':[], "angles":[], 'derivs':[], 'derivs2':[], 'derivs_num2':[]}
        for q in np.linspace(-np.pi, np.pi, 601):
            up = np.array([0, 0, 1])
            r = rotation_matrix(up, q)
            b = np.dot(r, a)
            ang, deriv, deriv_2 = vec_angle_derivs(a, b, up_vectors=up, order=2)
            data['rotations'].append(q)
            data['real_angles'].append(vec_angles(a, b, up_vectors=up)[0])
            data['angles'].append(ang.tolist())
            data['derivs'].append(deriv.tolist())
            data['derivs2'].append(deriv_2.tolist())

            data['derivs_num2'].append(fd(np.array([a, b])).derivative_tensor(1).tolist())

        import json
        with open(dump_file, 'w+') as f:
            json.dump(data, f)

    @debugTest
    def test_SetOps(self):

        unums, sorting = unique([1, 2, 3, 4, 5])
        self.assertEquals(unums.tolist(), [1, 2, 3, 4, 5])
        self.assertEquals(sorting.tolist(), [0, 1, 2, 3, 4])

        unums, sorting = unique([1, 1, 3, 4, 5])
        self.assertEquals(unums.tolist(), [1, 3, 4, 5])
        self.assertEquals(sorting.tolist(), [0, 1, 2, 3, 4])

        unums, sorting = unique([1, 3, 1, 1, 1])
        self.assertEquals(unums.tolist(), [1, 3])
        self.assertEquals(sorting.tolist(), [0, 2, 3, 4, 1])

        unums, sorting = unique([[1, 3], [1, 1], [1, 3]])
        self.assertEquals(unums.tolist(), [[1, 1], [1, 3]])
        self.assertEquals(sorting.tolist(), [1, 0, 2])

        inters, sortings, merge = intersection(
            [1, 1, 3, 2, 5],
            [0, 0, 0, 5, 1]
        )
        self.assertEquals(inters.tolist(), [1, 5])
        self.assertEquals(sortings[0].tolist(), [0, 1, 3, 2, 4])
        self.assertEquals(sortings[1].tolist(), [0, 1, 2, 4, 3])

        inters, sortings, merge = intersection(
            [
                [1, 3], [1, 1]
            ],
            [
                [1, 3], [0, 0]
            ]
        )
        self.assertEquals(inters.tolist(), [[1, 3]])
        self.assertEquals(sortings[0].tolist(), [1, 0])
        self.assertEquals(sortings[1].tolist(), [1, 0])

        diffs, sortings, merge = difference(
            [1, 1, 3, 2, 5],
            [0, 0, 0, 5, 1]
        )
        self.assertEquals(diffs.tolist(), [2, 3])
        self.assertEquals(sortings[0].tolist(), [0, 1, 3, 2, 4])
        self.assertEquals(sortings[1].tolist(), [0, 1, 2, 4, 3])

        diffs, sortings, merge = contained(
            [1, 1, 3, 2, 5],
            [0, 0, 0, 5, 1]
        )
        self.assertEquals(diffs.tolist(), [True, True, False, False, True])

    @debugTest
    def test_Sparse(self):

        shape = (1000, 100, 50)

        n_els = 10000
        np.random.seed(1)
        inds = tuple(np.random.choice(x, n_els) for x in shape)
        vals = np.random.rand(n_els)
        # remove dupes
        dupe_pos = np.where(
            np.logical_not(
                np.logical_and(
                    np.logical_and(
                        inds[0] == inds[0],
                        inds[1] == inds[1]
                    ),
                    inds[2] == inds[2]
                )
            )
        )
        inds = tuple(x[dupe_pos] for x in inds)
        vals = vals[dupe_pos]

        # `from_data` for backend flexibility
        array = SparseArray.from_data(
            (
                vals,
                inds
            ),
            shape=shape
        )

        self.assertEquals(array.shape, shape)
        block_vals, block_inds = array.block_data
        self.assertEquals(np.sort(block_vals).tolist(), np.sort(vals).tolist())
        for i in range(len(shape)):
            self.assertEquals(np.sort(block_inds[i]).tolist(), np.sort(inds[i]).tolist())

        woof = array[:, 1, 1] #type: SparseArray
        self.assertIs(type(woof), type(array))
        self.assertEquals(woof.shape, (shape[0],))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(np.logical_and(inds[1] == 1, inds[2] == 1))
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )


        with BlockProfiler('Sparse sampling', print_res=True):
            new_woof = array[:, 1, 1]  # type: SparseArray

