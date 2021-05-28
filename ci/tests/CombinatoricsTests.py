
from Peeves.TestUtils import *
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


    @debugTest
    def test_UniquePartitionPermutations(self):
        """
        Tests the generation of unique permutations of partitions
        :return:
        :rtype:
        """

        lens, parts = IntegerPartitioner.partitions(10, pad=True, return_lens=True)

        perms = UniquePermutations(parts[0]).get_permutations()
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
        perms = perm_builder.get_permutations(num_perms = 10)
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

        many_perms = perm_builder.get_permutations(initial_permutation=perms[8])
        self.assertEquals(len(many_perms), perm_builder.num_permutations - 8)

        perms = perm_builder.get_permutations(initial_permutation=perms[8], num_perms=10)
        self.assertEquals(many_perms[:10].tolist(), perms.tolist())
