
from Peeves.TestUtils import *
from Peeves import BlockProfiler
from unittest import TestCase
from McUtils.Combinatorics import *
import sys, os, numpy as np, itertools

class CombinatoricsTests(TestCase):

    @validationTest
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

        parts = IntegerPartitioner.partitions(10, pad=True, max_len=2)
        self.assertEquals(parts.tolist(), [[10, 0], [9, 1], [8, 2], [7, 3], [6, 4], [5, 5]])

        self.assertEquals(
            IntegerPartitioner.partition_indices([[10, 0], [9, 1], [8, 2], [7, 3], [6, 4], [5, 5]]).tolist(),
            list(range(6))
        )

        np.random.seed(0)
        full_stuff = IntegerPartitioner.partitions(10, pad=True, max_len=3)
        inds = np.random.choice(len(full_stuff), 5, replace=False)

        test_parts = full_stuff[inds]
        test_inds = IntegerPartitioner.partition_indices(test_parts)
        self.assertEquals(
            test_inds.tolist(),
            inds.tolist(),
            msg="{} should have indices {} but got {}".format(test_parts, inds, test_inds)
        )

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

    @debugTest
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

        test_set = [
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
        perm_builder = UniquePermutations(parts[15])
        perms = perm_builder.permutations(num_perms = 10)
        self.assertEquals(perms.tolist(), test_set)

        wat = UniquePermutations(parts[15]).permutations_from_indices(list(range(len(test_set))))
        # raise Exception(wat)
        self.assertEquals(wat.tolist(), test_set)

        inds = perm_builder.index_permutations(perms)
        self.assertEquals(inds.tolist(), list(range(len(inds))))

        many_perms = perm_builder.permutations(initial_permutation=perms[8])
        self.assertEquals(len(many_perms), perm_builder.num_permutations - 8)

        perms = perm_builder.permutations(initial_permutation=perms[8], num_perms=10)
        self.assertEquals(many_perms[:10].tolist(), perms.tolist())

        perm_builder = UniquePermutations([3, 1, 1, 0, 0])

        test_set = [
            [3, 0, 0, 1, 1],
            [1, 1, 3, 0, 0],
            [0, 3, 1, 1, 0]
        ]
        inds = perm_builder.index_permutations(test_set)
        all_perms = perm_builder.permutations()
        all_list_perms = all_perms.tolist()
        test_inds = [all_list_perms.index(x) for x in test_set]

        self.assertEquals(inds.tolist(), test_inds)

        np.random.seed(0)
        test_inds = np.random.choice(len(all_perms), 25, replace=False)
        test_set = all_perms[test_inds,]
        sorting = np.lexsort(-np.flip(test_set, axis=1).T)[:5]
        test_set = test_set[sorting,]
        test_inds = test_inds[sorting,]

        # print("="*50)
        # print(test_set)
        inds = perm_builder.index_permutations(test_set)
        self.assertEquals(inds.tolist(), test_inds.tolist())

        # print('=' * 50)
        perms = perm_builder.permutations_from_indices(inds)
        self.assertEquals(perms.tolist(), test_set.tolist())

    @validationTest
    def test_IntegerPartitionPermutations(self):
        """
        Tests generating and indexing integer partition permutations
        :return:
        :rtype:
        """

        np.random.seed(0)
        part_perms = IntegerPartitionPermutations(5)

        full_stuff = np.concatenate(part_perms.get_partition_permutations(), axis=0) # build everything so we can take subsamples to index

        inds = np.random.choice(len(full_stuff), 8, replace=False)
        subperms = full_stuff[inds,]

       #  """Exception: (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       # 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       # 35, 36, 37, 38, 39, 40]), 41)"""
       #  """
       #  (array([40, 39, 37, 34, 29, 38, 36, 33, 28, 32, 27, 22, 21, 35, 31, 26, 20,
       # 25, 19, 13, 18, 12, 30, 24, 17, 16, 11, 10,  5, 23, 15,  9,  8,  4,
       # 14,  7,  3,  6,  2,  1]), 40)
       # """

        # with BlockProfiler(name='perm method'):
        perm_inds = part_perms.get_partition_permutation_indices(subperms)
        self.assertEquals(inds.tolist(), perm_inds.tolist())