
from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Combinatorics import *
import sys, os, numpy as np, itertools

class CombinatoricsTests(TestCase):

    @debugTest
    def test_IntegerPartitions(self):
        """
        Tests integer partitioning algs
        """

        num_parts = IntegerPartitioner.count_partitions(3)
        self.assertEquals(num_parts, 3)

        n = np.array([3])
        M = n
        l = n
        num_parts = IntegerPartitioner.count_partitions(n, M, l)
        self.assertEquals(num_parts, [3])

        # num_parts = IntegerPartitioner.count_partitions(6, 2, 3)
        # self.assertEquals(num_parts, 1)

        # num_parts = IntegerPartitioner.count_partitions(5, 3, 3)
        # self.assertEquals(num_parts, 3)

        n = np.array([3, 5, 2, 6])
        M = np.array([1, 3, 1, 2])
        l = np.array([3, 3, 3, 3])
        # raise Exception([
        #     IntegerPartitioner.count_partitions(3, 1, 3),
        #     IntegerPartitioner.count_partitions(5, 3, 3),
        #     IntegerPartitioner.count_partitions(2, 1, 3),
        #     IntegerPartitioner.count_partitions(6, 2, 3)
        # ])
        num_parts = IntegerPartitioner.count_partitions(n, M, l)
        self.assertEquals(num_parts.tolist(), [1, 3, 1, 1])

        n = np.array([3, 5, 2, 6, 10, 10])
        M = np.array([1, 3, 1, 2, 10,  5])
        l = np.array([3, 3, 3, 3,  3,  3])
        # raise Exception([
        #     IntegerPartitioner.count_partitions(3, 1, 3),
        #     IntegerPartitioner.count_partitions(5, 3, 3),
        #     IntegerPartitioner.count_partitions(2, 1, 3),
        #     IntegerPartitioner.count_partitions(6, 2, 3)
        # ])
        num_parts = IntegerPartitioner.count_partitions(n, M, l)
        self.assertEquals(num_parts.tolist(), [1, 3, 1, 1, 14, 5])

        num_greater = IntegerPartitioner.count_exact_length_partitions_in_range(4, 4, 2, 2)
        len2_4s = IntegerPartitioner.partitions(4, max_len=2)[1]
        raw_counts = len([x for x in len2_4s if len(x) == 2 and x[0] > 2])
        self.assertEquals(num_greater, raw_counts)


        parts = IntegerPartitioner.partitions(3)
        self.assertEquals([p.tolist() for p in parts], [ [[3]], [[2, 1]], [[1, 1, 1]]])

        parts = IntegerPartitioner.partitions(3, pad=True)
        self.assertEquals(parts.tolist(), [[3, 0, 0], [2, 1, 0], [1, 1, 1]])

        parts = IntegerPartitioner.partitions(3, pad=True, max_len=2)
        self.assertEquals(parts.tolist(), [[3, 0], [2, 1]])

        num_parts = IntegerPartitioner.count_partitions(10)
        self.assertEquals(num_parts, 42)

        parts = IntegerPartitioner.partitions(5)
        self.assertEquals([p.tolist() for p in parts], [
            [[5]],
            [[4, 1], [3, 2]],
            [[3, 1, 1], [2, 2, 1]],
            [[2, 1, 1, 1]],
            [[1, 1, 1, 1, 1]]
        ])

        # parts = IntegerPartitioner.partitions(10, pad=True, max_len=2)
        # self.assertEquals(parts.tolist(), [[10, 0], [9, 1], [8, 2], [7, 3], [6, 4], [5, 5]])
        #
        # self.assertEquals(
        #     IntegerPartitioner.partition_indices([[10, 0], [9, 1], [8, 2], [7, 3], [6, 4], [5, 5]]).tolist(),
        #     list(range(6))
        # )
        #
        # np.random.seed(0)
        # full_stuff = IntegerPartitioner.partitions(10, pad=True, max_len=3)
        # inds = np.random.choice(len(full_stuff), 5, replace=False)
        #
        # test_parts = full_stuff[inds]
        # test_inds = IntegerPartitioner.partition_indices(test_parts)
        # self.assertEquals(
        #     test_inds.tolist(),
        #     inds.tolist(),
        #     msg="{} should have indices {} but got {}".format(test_parts, inds, test_inds)
        # )

        np.random.seed(1)
        full_stuff = IntegerPartitioner.partitions(10, pad=True, max_len=7)

        inds = np.random.choice(len(full_stuff), 35, replace=False)

        test_parts = full_stuff[inds]
        # raise Exception(inds, test_parts, full_stuff)
        test_inds = IntegerPartitioner.partition_indices(test_parts)
        self.assertEquals(
            test_inds.tolist(),
            inds.tolist(),
            msg="{} should have indices {} but got {}".format(test_parts, inds, test_inds)
        )


    @validationTest
    def test_UniquePartitionPermutations(self):
        """
        Tests the generation of unique permutations of partitions
        :return:
        :rtype:
        """

        lens, parts = IntegerPartitioner.partitions(10, pad=True, return_lens=True)

        perms = UniquePermutations(parts[0]).permutations()
        self.assertEquals(perms.tolist(),
                          [
                              [10,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                              [ 0, 10,  0,  0,  0,  0,  0,  0,  0,  0],
                              [ 0,  0, 10,  0,  0,  0,  0,  0,  0,  0],
                              [ 0,  0,  0, 10,  0,  0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0, 10,  0,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  0, 10,  0,  0,  0,  0],
                              [ 0,  0,  0,  0,  0,  0, 10,  0,  0,  0],
                              [ 0,  0,  0,  0,  0,  0,  0, 10,  0,  0],
                              [ 0,  0,  0,  0,  0,  0,  0,  0, 10,  0],
                              [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 10]
                          ]
                          )

        perm_builder = UniquePermutations(parts[15])
        perms = perm_builder.permutations(num_perms = 10)
        self.assertEquals(perms.tolist(),
                          [
                              [6, 2, 1, 1, 0, 0, 0, 0, 0, 0],
                              [6, 2, 1, 0, 1, 0, 0, 0, 0, 0],
                              [6, 2, 1, 0, 0, 1, 0, 0, 0, 0],
                              [6, 2, 1, 0, 0, 0, 1, 0, 0, 0],
                              [6, 2, 1, 0, 0, 0, 0, 1, 0, 0],
                              [6, 2, 1, 0, 0, 0, 0, 0, 1, 0],
                              [6, 2, 1, 0, 0, 0, 0, 0, 0, 1],
                              [6, 2, 0, 1, 1, 0, 0, 0, 0, 0],
                              [6, 2, 0, 1, 0, 1, 0, 0, 0, 0],
                              [6, 2, 0, 1, 0, 0, 1, 0, 0, 0]
                          ]
                          )

        inds = perm_builder.index_permutations(perms)
        self.assertEquals(inds.tolist(), list(range(len(inds))))

        many_perms = perm_builder.permutations(initial_permutation=perms[8])
        self.assertEquals(len(many_perms), perm_builder.num_permutations - 8)

        perms = perm_builder.permutations(initial_permutation=perms[8], num_perms=10)
        self.assertEquals(many_perms[:10].tolist(), perms.tolist())

    @inactiveTest
    def test_IntegerPartitionPermutations(self):
        """
        Tests generating and indexing integer partition permutations
        :return:
        :rtype:
        """

        np.random.seed(0)
        part_perms = IntegerPartitionPermutations(5)

        full_stuff = np.concatenate(part_perms.get_partition_permutations(), axis=0) # build everything so we can take subsamples to index
        inds = np.random.choice(len(full_stuff), 15, replace=False)

        subperms = full_stuff[inds,]
        perm_inds = part_perms.get_partition_permutation_indices(subperms)

        raise Exception(perm_inds)