"""
Utilities for working with permutations and permutation indexing
"""

import numpy as np
import collections, functools as ft

__all__ = [
    "IntegerPartitioner",
    "UniquePermutations",
    "IntegerPartitionPermutations",
    "SymmetricGroupGenerator"
]


class IntegerPartitioner:

    def __init__(self):
        raise NotImplementedError('{} is a singleton class'.format(
            type(self).__name__
        ))

    _partition_counts = None
    @classmethod
    def _manage_counts_array(cls, n, M, l):
        grew = False
        if cls._partition_counts is None:
            grew = True
            # we just initialize this 3D array to be the size we need
            cls._partition_counts = np.full((n, M, l), -1, dtype=int)

        if n > cls._partition_counts.shape[0]:
            grew = True
            # grow in size along this axis
            # we grow by 2X the amount we really need to
            # so as to amortize the cost of the concatenations (thank you Lyle)
            ext_shape = (
                            2 * (n - cls._partition_counts.shape[0]),
                        ) + cls._partition_counts.shape[1:]
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.full(ext_shape, -1, dtype=cls._partition_counts.dtype)
            ],
                axis=0
            )
        if M > cls._partition_counts.shape[1]:
            grew = True
            ext_shape = (
                cls._partition_counts.shape[0],
                2 * (M - cls._partition_counts.shape[1]),
                cls._partition_counts.shape[2]
            )
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.full(ext_shape, -1, dtype=cls._partition_counts.dtype)
            ],
                axis=1
            )
        if l > cls._partition_counts.shape[2]:
            grew = True
            ext_shape = (
                cls._partition_counts.shape[0],
                cls._partition_counts.shape[1],
                2 * (M - cls._partition_counts.shape[2])
            )
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.full(ext_shape, -1, dtype=cls._partition_counts.dtype)
            ],
                axis=2
            )

        if grew:
            for i in range(cls._partition_counts.shape[0]):
                cls._partition_counts[i, i:, 0] = 1
                cls._partition_counts[i, 0, i:] = 1

    @classmethod
    def count_partitions(cls, n, M=None, l=None, manage_counts=True):
        """
        Uses the recurrence relation written out here
        https://en.wikipedia.org/wiki/Partition_(number_theory)#Partitions_in_a_rectangle_and_Gaussian_binomial_coefficients
        We cache the terms as a 2D list-of-lists because we don't need this
        part of the code to be blazingly fast but would like repeats to not
        do unnecessary work (and because the memory cost is small...)

        :param n:
        :type n:
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """

        if isinstance(n, (int, np.integer)):
            # just simple checks that could cause stuff to break or be overly slow otherwise
            if l is None:
                l = n
            if M is None:
                M = n
            if l > n:
                l = n
            if M > n:
                M = n

            if n == 0 and M ==0 and l == 0:
                return 1
            if M <= 0 or l <= 0:
                return 0
            if l == 1 and M < n:
                return 0

            if manage_counts:
                cls._manage_counts_array(n, M, l)

            counts = cls._partition_counts[n-1, M-1, l-1]
            if counts < 0:
                # print("======>", (n, M, l))
                t1 = cls.count_partitions(n, M, l-1, manage_counts=False)
                t2 = cls.count_partitions(n-l, M-1, l, manage_counts=False)
                counts = t1 + t2
                # print(((n, M, l-1), t1), ((n-l, M-1, l), t2))
                cls._partition_counts[n-1, M-1, l-1] = counts
        else:
            # we assume we got a numpy array of equal length for each term
            n = np.asanyarray(n)#.astype(int)
            M = np.asanyarray(M)#.astype(int)
            l = np.asanyarray(l)#.astype(int)

            too_long = np.where(l > n)
            if len(too_long) > 0:
                too_long = too_long[0]
                if len(too_long) > 0:
                    l = l.copy()
                    l[too_long] = n[too_long]

            too_big = np.where(M > n)
            if len(too_big) > 0:
                too_big = too_big[0]
                if len(too_big) > 0:
                    M = M.copy()
                    M[too_big] = n[too_big]

            if manage_counts:
                cls._manage_counts_array(np.max(n), np.max(M), np.max(l))

            # print(n.dtype, M.dtype, l.dtype)
            counts = cls._partition_counts[n - 1, M - 1, l - 1]

            zero_lens = l == 0
            zero_size = M == 0
            should_be_0 = np.where(ft.reduce(np.logical_or,
                                             [
                                                 zero_size, zero_lens,
                                                 l < 0,
                                                 M < 0,
                                                 np.logical_and(l == 1, M < n)
                                             ]))
            if len(should_be_0) > 0:
                should_be_0 = should_be_0[0]
                if len(should_be_0) > 0:
                    counts[should_be_0] = 0

            should_be_1 = np.where(ft.reduce(np.logical_and, [n == 0, zero_size, zero_lens]))
            if len(should_be_1) > 0:
                should_be_1 = should_be_1[0]
                if len(should_be_1) > 0:
                    counts[should_be_1] = 1

            needs_updates = np.where(counts < 0)

            if len(needs_updates) > 0:
                needs_updates = needs_updates[0]
                if len(needs_updates) > 0:
                    # wtf
                    n = n[needs_updates]
                    M = M[needs_updates]
                    l = l[needs_updates]
                    t1 = cls.count_partitions(n, M, l - 1, manage_counts=False)
                    t2 = cls.count_partitions(n - l, M - 1, l, manage_counts=False)
                    counts[needs_updates] = t1 + t2
                    cls._partition_counts[n - 1, M - 1, l - 1] = counts[needs_updates]

        return counts

    @classmethod
    def count_exact_length_partitions(cls, n, M, l):
        """
        Unexpectedly common thing to want and a non-obvious formula

        :param n:
        :type n:
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """
        # print((n, M, l), ">>", (n - l, M - 1, l))
        return cls.count_partitions(n - l, M - 1, l)

    @classmethod
    def count_exact_length_partitions_in_range(cls, n, m, M, l):
        """
        Returns the partitions with  k > M but length exactly L

        :param n:
        :type n:
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """
        wat1 = cls.count_exact_length_partitions(n, m, l)
        wat2 = cls.count_exact_length_partitions(n, M, l)
        # print(wat1, wat2)
        return wat1 - wat2

    # @classmethod
    # def count_
    #     return cls.count_partitions(sums - counts, sums - 1, counts) - cls.count_partitions(sums - counts, parts[:, i] - 1,
    #                                                                                  counts)

    @classmethod
    def partitions(cls, n, pad=False, return_lens = False, max_len=None):
        """
        Returns partitions in descending lexicographic order
        Adapted from Kelleher to return terms ordered by length and then second in descending
        lex order which while a computationally suboptimal is very natural for a mapping onto
        physical phenomena (and also it's easier for storage)

        :param n: integer to partition
        :type n: int
        :param return_len: whether to return the length or not
        :type return_len:
        :return:
        :rtype:
        """

        if max_len is None or max_len > n:
            max_len = n
        l = max_len

        # total_partitions = cls.count_partitions(n)
        count_totals = np.array([cls.count_partitions(n, l=i+1) for i in range(l)])
        counts = np.concatenate([count_totals[:1], np.diff(count_totals)], axis=0)
        # count_totals = np.flip(count_totals)
        if pad:
            storage = np.zeros((count_totals[-1], l), dtype=int)
        else:
            storage = [np.zeros((c, i+1), dtype=int) for i,c in enumerate(counts)]
            # raise Exception([s.shape for s in storage], counts, count_totals)

        increments = np.zeros(l, dtype=int)

        partition = np.ones(n, dtype=int)
        partition[0] = n
        k = q = 0 # k tracks the length of the permutation and q tracks where we're writing
        if pad:
            storage[increments[k], :k+1] = partition[:k+1]
        else:
            storage[k][increments[k]] = partition[:k+1]
        increments[k] += 1
        for _ in range(1, cls.count_partitions(n)):
            # everytime we reach this line we have completed a "transition"
            # where we convert an element like a -> a - 1, 1
            # the number of such transitions is hard to calculate a priori?
            if partition[q] == 2:
                # this is a special optimized case where, since the array is initialized to 1
                # when the element we're trying to "transition" is equal to 2 we just set it back to 1
                # and increment the length of our partition
                k += 1
                partition[q] = 1
                q -= 1
            else:
                # perform the transition, decrementing the term at position q
                # and extending the array where necessary
                m = partition[q] - 1
                n1 = k - q + 1
                partition[q] = m
                if n1 >= m:
                    for n1 in range(n1, m-1, -m):
                        q += 1
                        partition[q] = m
                    n1 -= m
                if n1 == 0:
                    k = q
                else:
                    k = q + 1
                    if n1 > 1:
                        q += 1
                        partition[q] = n1

            if k < l-1:
                if pad:
                    storage[count_totals[k-1]+increments[k], :k+1] = partition[:k+1]
                else:
                    storage[k][increments[k]] = partition[:k+1]
                increments[k] += 1
            elif k == l-1:
                if pad:
                    storage[count_totals[k-1]+increments[k], :k+1] = partition[:k+1]
                else:
                    storage[k][increments[k]] = partition[:k+1]
                increments[k] += 1
                if increments[k] == counts[k]:
                    break

        if return_lens:
            if pad:
                lens = np.ones(count_totals[-1], dtype=int)
                for i, bounds in enumerate(zip(counts, counts[1:])):
                    a, b = bounds
                    lens[a:b] = i + 1
            else:
                lens = [np.full(c, i + 1, dtype=int) for i, c in enumerate(counts)]
            return lens, storage
        else:
            return storage

    @classmethod
    def partition_indices(cls, parts, sums=None, counts=None):
        """
        Provides a somewhat quick way to get the index of a set of
        integer partitions.
        Parts must be padded so that all parts are the same length.
        If the sums of the partitions are known ahead of time they may be passed
        Similarly if the numbers of non-zero elements in the partitions are known
        ahead of time they may _also_ be passed

        :param parts:
        :type parts: np.ndarray
        :param sums:
        :type sums: np.ndarray
        :return:
        :rtype:
        """

        parts = np.asanyarray(parts).astype(int)
        smol = parts.ndim == 1
        if smol:
            parts = parts.reshape((1, parts.shape[0]))

        if sums is None:
            sums = np.sum(parts, axis=1)

        # need this to track
        if counts is None:
            counts = np.count_nonzero(parts, axis=1)

        # num_before = np.zeros(parts.shape[0], dtype=int) # where the indices will be in reverse-lex order
        inds = np.arange(parts.shape[0])
        num_before = cls.count_partitions(sums, sums, counts - 1)

        # print(np.column_stack([sums, parts, num_before]))
        for i in range(np.max(counts) - 1): # exhaust all elements except the last one where contrib will always be zero
            # now we need to figure out where the counts are greater than 1
            mask = np.where(counts > 1)[0]
            if len(mask) > 0:
                counts = counts[mask]
                inds = inds[mask]
                sums = sums[mask]
                parts = parts[mask]
                if i > 0:
                    subsums = cls.count_exact_length_partitions_in_range(sums, parts[:, i-1], parts[:, i], counts)
                else:
                    subsums = cls.count_exact_length_partitions_in_range(sums, sums, parts[:, i], counts)
                #cls.count_partitions(sums - counts, sums - 1, counts) - cls.count_partitions(sums - counts, parts[:, i] - 1, counts)
                num_before[inds] += subsums
                counts -= 1
                sums -= parts[:, i]

        inds = num_before
        if smol:
            inds = inds[0]

        return inds

class UniquePermutations:
    """
    Provides permutations for a _single_ integer partition (very important)
    Also provides a classmethod interface to support the case
    where we don't want to instantiate a permutations object for every partition
    """
    def __init__(self, partition):
        self.part = np.flip(np.sort(partition))
        v, c = np.unique(partition, return_counts=True)
        self.vals = np.flip(v)
        self.counts = np.flip(c)
        self.dim = len(partition)
        self._num = None

    @classmethod
    def get_permutation_class_counts(cls, partition, sort_by_counts=False):
        v, c = np.unique(partition, return_counts=True)
        if sort_by_counts:
            s = np.argsort(c)
            v = v[s]
            c = c[s]
        else:
            v, c = np.flip(v), np.flip(c)
        return v, c

    @property
    def num_permutations(self):
        """
        Counts the number of unique permutations of the partition
        :param counts:
        :type counts:
        :return:
        :rtype:
        """
        if self._num is None:
            self._num = self.count_permutations(self.counts)
        return self._num

    @staticmethod
    def count_permutations(counts):
        """
        Counts the number of unique permutations of the given "counts"
        which correspond to the number of nodes in the unique permutation tree
        :param counts:
        :type counts:
        :return:
        :rtype:
        """
        import math

        subfac = np.prod([math.factorial(x) for x in counts])
        ndim_fac = math.factorial(np.sum(counts))

        return ndim_fac // subfac

    def permutations(self, initial_permutation=None, return_indices=False, num_perms=None):
        """
        Returns the permutations of the input array
        :param initial_permutation:
        :type initial_permutation:
        :param return_indices:
        :type return_indices:
        :param classes:
        :type classes:
        :param counts:
        :type counts:
        :param num_perms:
        :type num_perms:
        :return:
        :rtype:
        """

        if initial_permutation is None:
            initial_permutation = self.part
            if num_perms is None:
                num_perms = self.num_permutations
        else:
            if num_perms is None:
                num_perms = self.num_permutations - self.index_permutations(initial_permutation)

        return self.get_subsequent_permutations(initial_permutation, return_indices=return_indices, num_perms=num_perms)

    @classmethod
    def get_subsequent_permutations(cls, initial_permutation, return_indices=False, classes=None, counts=None, num_perms=None):
        """
        Returns the permutations of the input array
        :return:
        :rtype:
        """

        initial_permutation = np.asanyarray(initial_permutation)
        if num_perms is None:
            # need to determine where we start/how far we have to go
            if counts is None or classes is None:
                classes, counts = np.unique(initial_permutation, return_counts=True)
            total_perms = cls.count_permutations(counts)
            skipped = cls.get_permutation_indices(initial_permutation, classes, counts, num_permutations=total_perms)
            num_perms = total_perms - skipped

        dim = len(initial_permutation)

        storage = np.zeros((num_perms, dim), dtype=initial_permutation.dtype)
        if return_indices:
            inds = np.zeros((num_perms, dim), dtype=int) # this could potentially be narrower
        else:
            inds = None

        part = initial_permutation.copy()
        cls._fill_permutations_direct(storage, inds, part, dim)

        if return_indices:
            return inds, storage
        else:
            return storage

    @classmethod
    def _fill_permutations_direct(cls, storage, inds, partition, dim):
        """
        Builds off of this algorithm for generating permutations
        in lexicographic order: https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
        Then we adapt it so that it works in _reverse_ lex order since that's how our partitions come in
        This adaption is done just by pretending the the numbers are all negated so all ordering relations
        flip

        :param storage:
        :type storage:
        :param inds:
        :type inds:
        :return:
        :rtype:
        """

        swap = np.arange(len(partition))
        for i in range(len(storage)):
            storage[i] = partition
            if inds is not None:
                inds[i] = swap

            # find largest index such that the next element in the
            # partition is smaller (i.e. find where we need to do our next swap)
            # I'd like to do this with numpy builtins instead of a loop
            # or maybe some partial ordering approach or something
            # but don't have it quite figured out yet
            for i in range(dim-2, -1, -1):
                if partition[i] > partition[i+1]:
                    break
            else:
                break

            # find the next-smallest index such that
            # the partition element is smaller than the one at the swap
            # position
            for j in range(dim-1, i, -1):
                if partition[i] > partition[j]:
                    break

            # swap the terms and reverse the sequence of elements leading
            # up to it
            partition[(i, j),] = partition[(j, i),]
            partition[i+1:] = np.flip(partition[i+1:])

            if inds is not None:
                swap[(i, j),] = swap[(j, i),]
                swap[i+1:] = np.flip(swap[i+1:])

        return storage

    @staticmethod
    def _subtree_counts(total, ndim, counts, where):
        """
        Computes the number of states in the tree built from decrementing counts[where] by 1
        Is it trivially simple? Yes
        But there's power to having it be named.
        :param total:
        :type total:
        :param ndim:
        :type ndim:
        :param counts:
        :type counts:
        :param where:
        :type where:
        :return:
        :rtype:
        """
        mprod = total * counts[where]
        if mprod % ndim != 0:
            raise ValueError("subtree counts {} don't comport with dimension {}".format(
                mprod, ndim
            ))
        return mprod // ndim

    @staticmethod
    def _reverse_subtree_counts(subtotal, ndim, counts, j):
        """
        Given subtotal = (total * counts[j]) // ndim
              total    = (subtotal * ndim) // counts[j]
        :return:
        :rtype:
        """

        return (subtotal * ndim) // counts[j]

    def index_permutations(self, perms, assume_sorted=False):
        """
        Gets permutations indices assuming all the data matches the held stuff
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        return self.get_permutation_indices(self.vals, self.counts, perms,
                                            assume_sorted=assume_sorted, dim=self.dim, num_permutations=self.num_permutations)

    # @classmethod
    # def get_next_permutation_from_prev(cls, classes, counts, class_map,
    #                                    ndim,
    #                                    cur,
    #                                    prev, prev_index,
    #                                    prev_dim,
    #                                    subtree_counts,
    #                                    ):
    #     """
    #     Pulls the next index by reusing as much info as possible from
    #     previous index
    #     Less able to be efficient than computing many indices at once so prefer that if
    #     possible
    #
    #     :return:
    #     :rtype:
    #     """
    #
    #     if prev is not None:
    #         diffs = np.not_equal(cur, prev)
    #         # we reuse as much work as we can by only backtracking where we need to
    #         agree_pos = np.where(diffs)[0][0]  # where the first disagreement occurs...I'd like this to be less inefficient
    #         num_diff = ndim - agree_pos  # number of differing states
    #         if num_diff == 0:  # same state so just reuse the previous value
    #             return prev_index, prev_dim
    #
    #         elif agree_pos == 0:
    #             # no need to actually backtrack when we know what we're gonna get
    #             cur_dim = ndim - 1
    #         else:
    #             # at this point cur_dim gives us the number of trailing
    #             # digits that are equivalent in the previous permutation
    #             # so we only need to back-track to where the new state begins to
    #             # differ from the old one,
    #             for i in range(ndim - prev_dim - 2, agree_pos - 1, -1):
    #                 j = class_map[prev[i]]
    #                 counts[j] += 1
    #                 # tree_data[cur_dim, 1] = 0
    #                 # tree_data[cur_dim, 0] = 0
    #                 cur_dim += 1
    #             # I'm not sure why this didn't break earlier without this...
    #             tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]
    #
    #             # print(ndim-agree_pos)
    #             # print("<<", cur_dim, counts, tree_data[:, 1], tree_data[:, 0])
    #             state = state[agree_pos:]
    #
    #     # we loop through the elements in the permutation and
    #     # add up number of elements in the subtree that would precede
    #     # the state in reverse-lexicographic order
    #     for i, el in enumerate(state):
    #         for j in range(nterms):
    #             if counts[j] == 0:
    #                 continue
    #             # print("  ", cur_dim, counts, tree_data[:, 1], tree_data[:, 0])
    #             subtotal = cls._subtree_counts(tree_data[cur_dim, 1], cur_dim + 1, counts, j)
    #             if classes[j] == el:
    #                 cur_dim -= 1
    #                 counts[j] -= 1
    #                 tree_data[cur_dim, 1] = subtotal
    #                 tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]
    #                 break
    #             else:
    #                 tree_data[cur_dim, 0] += subtotal
    #
    #         # short circuit if we've gotten down to a terminal node where
    #         # there is just one unique element
    #         if tree_data[cur_dim, 1] == 1:
    #             # print(">>", cur_dim, counts, tree_data[:, 1], tree_data[:, 0])
    #             break
    #
    #     inds[sn] = tree_data[cur_dim, 0]


    @classmethod
    def get_permutation_indices(cls, perms, classes=None, counts=None, assume_sorted=False, dim=None, num_permutations=None):
        """
        Classmethod interface to get indices for permutations
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        perms = np.asanyarray(perms)

        if classes is None or counts is None:
            classes, counts = cls.get_permutation_class_counts(perms[0])

        smol = perms.ndim == 1
        if smol:
            assume_sorted = True
            perms = np.reshape(perms, (1, len(perms)))

        if not assume_sorted:
            sorting = np.lexsort(-np.flip(perms, axis=1).T)
            perms = perms[sorting,]
        else:
            sorting = None

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        tree_data = np.zeros((dim, 2), dtype=int)
        cur_dim = dim - 1
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        tree_data[cur_dim, 1] = num_permutations
        ndim = dim
        # we make a constant-time lookup for what a value maps to in
        # terms of position in the counts array
        class_map = {v:i for i,v in enumerate(classes)}
        init_counts = counts
        counts = np.copy(counts) # we're going to modify this in-place
        nterms = len(counts)
        # stack = collections.deque()  # stack of current data to reuse

        # determine where each successive permutation differs so we can know how much to reuse
        diffs = np.not_equal(perms[:-1], perms[1:])

        inds = np.full((len(perms),), -1)
        for sn, state in enumerate(perms):
            if sn > 0:
                # we reuse as much work as we can by only backtracking where we need to
                agree_pos = np.where(diffs[sn-1])[0] # where the first disagreement occurs...I'd like this to be less inefficient
                if len(agree_pos) == 0:
                    agree_pos = ndim
                else:
                    agree_pos = agree_pos[0]
                num_diff = ndim - agree_pos  # number of differing states
                if num_diff == 0:  # same state so just reuse the previous value
                    if inds[sn - 1] == -1:
                        raise ValueError("permutation {} tried to reused bad value from permutation {}".format(
                            perms[sn], perms[sn - 1]
                        ))
                    inds[sn] = inds[sn - 1]
                    continue
                elif agree_pos == 0:
                    # no need to actually backtrack when we know what we're gonna get
                    cur_dim = ndim - 1
                    tree_data[cur_dim, 1] = num_permutations
                    tree_data[cur_dim, 0] = 0
                    counts = init_counts.copy()
                else:
                    prev = perms[sn-1]
                    # at this point cur_dim gives us the number of trailing
                    # digits that are equivalent in the previous permutation
                    # so we only need to back-track to where the new state begins to
                    # differ from the old one,
                    for i in range(ndim - cur_dim - 2, agree_pos-1, -1):
                        j = class_map[prev[i]]
                        counts[j] += 1
                        # tree_data[cur_dim, 1] = 0
                        # tree_data[cur_dim, 0] = 0
                        cur_dim += 1
                    # I'm not sure why this didn't break earlier without this...
                    tree_data[cur_dim, 0] = tree_data[cur_dim+1, 0]

                    state = state[agree_pos:]

            # we loop through the elements in the permutation and
            # add up number of elements in the subtree that would precede
            # the state in reverse-lexicographic order
            for i, el in enumerate(state):
                for j in range(nterms):
                    if counts[j] == 0:
                        continue
                    subtotal = cls._subtree_counts(tree_data[cur_dim, 1], cur_dim+1, counts, j)
                    if classes[j] == el:
                        cur_dim -= 1
                        counts[j] -= 1
                        tree_data[cur_dim, 1] = subtotal
                        tree_data[cur_dim, 0] = tree_data[cur_dim+1, 0]
                        break
                    else:
                        tree_data[cur_dim, 0] += subtotal

                # short circuit if we've gotten down to a terminal node where
                # there is just one unique element
                if tree_data[cur_dim, 1] == 1:
                    break

            inds[sn] = tree_data[cur_dim, 0]

        if sorting is not None:
            inds = inds[np.argsort(sorting)]
        elif smol:
            inds = inds[0]

        return inds

    @classmethod
    def get_permutations_from_indices(cls, classes, counts, indices, assume_sorted=False,
                                      dim=None, num_permutations=None, check_indices=True, no_backtracking=False):
        """
        Classmethod interface to get permutations given a set of indices
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        smol = isinstance(indices, (int, np.integer))
        if smol:
            indices = [indices]
        indices = np.asanyarray(indices)

        if not assume_sorted:
            sorting = np.argsort(indices)
            indices = indices[sorting,]
        else:
            sorting = None

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        tree_data = np.zeros((dim, 2), dtype=int)
        depth = 0 # where we're currently writing
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        if check_indices:
            bad_spots = np.where(indices >= num_permutations)[0]
            if len(bad_spots) > 0:
                raise ValueError("Classes/counts {}/{} only supports {} permutations. Can't return permutations {}".format(
                    classes, counts, num_permutations,
                    indices[bad_spots]
                ))

        tree_data[depth, 1] = num_permutations
        # we make a constant-time lookup for what a value maps to in
        # terms of position in the counts array
        class_map = {v:i for i,v in enumerate(classes)}
        init_counts = counts
        counts = np.copy(counts) # we're going to modify this in-place
        nterms = len(counts)

        perms = np.zeros((len(indices), dim), dtype=int)

        for sn, idx in enumerate(indices):

            # we back track directly to where the sum of the subtotal and the current num_before
            # is greater than the target index
            if no_backtracking:

                # so we can make sure a non-backtracking solution still works
                depth = 0
                counts = init_counts.copy()
                tree_data[depth, 1] = num_permutations
                tree_data[depth, 0] = 0

            elif sn > 0:
                tree_sums = tree_data[:depth, 0] + tree_data[1:depth+1, 1]
                target_depth = np.where(tree_sums > idx)[0]
                if len(target_depth) > 0:
                    target_depth = np.max(target_depth)
                    # backtracks = len(tree_sums) - 1 - np.max()
                    # target_depth = depth - backtracks
                    prev = perms[sn-1]
                    for d in range(depth - 1, target_depth, -1):
                        inc_el = prev[d]
                        j = class_map[inc_el]
                        depth -= 1
                        counts[j] += 1
                    perms[sn, :depth] = prev[:depth]
                    tree_data[depth, 0] = tree_data[depth-1, 0]
                else:
                    # means we need to backtrack completely
                    # so why even walk?
                    depth = 0
                    counts = init_counts.copy()
                    tree_data[depth, 1] = num_permutations
                    tree_data[depth, 0] = 0

            done = False
            for i in range(depth, dim): # we only need to do at most cur_dim writes
                # We'll get each element 1-by-1 in an O(d) fashion.
                # This isn't blazingly fast but it'll work okay

                # loop over the classes of elements and see at which point dropping an element exceeds the current index
                # which tells is that the _previous_ term was the correct one
                for j in range(nterms):
                    if counts[j] == 0:
                        continue

                    subtotal = cls._subtree_counts(tree_data[depth, 1], dim - depth, counts, j)
                    test = tree_data[depth, 0] + subtotal
                    if test > idx: # or j == nterms-1: there's got to be _some_ index at which we get past idx I think...
                        depth += 1
                        counts[j] -= 1
                        perms[sn, i] = classes[j]
                        tree_data[depth, 1] = subtotal
                        tree_data[depth, 0] = tree_data[depth-1, 0]
                        if tree_data[depth, 0] == idx:
                            # we know that every next iteration will _also_ do an insertion
                            # so we can just do that all at once
                            insertion = np.concatenate(
                                [
                                    np.full(counts[l], classes[l], dtype=perms.dtype)
                                    for l in range(nterms) if counts[l] > 0
                                 ]
                            )
                            perms[sn, i + 1:] = insertion
                            done = True
                        break
                    else:
                        tree_data[depth, 0] = test

                if done:
                    break

        if sorting is not None:
            perms = perms[np.argsort(sorting)]
        elif smol:
            perms = perms[0]

        return perms

    def permutations_from_indices(self, indices, assume_sorted=False):
        """
        Gets permutations indices assuming all the data matches the held stuff
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        return self.get_permutations_from_indices(self.vals, self.counts, indices,
                                            assume_sorted=assume_sorted, dim=self.dim, num_permutations=self.num_permutations)

    @classmethod
    def get_standard_permutation(cls, counts, classes):
        return np.concatenate(
                                [np.full(counts[l], classes[l], dtype=classes.dtype) for l in range(len(counts)) if counts[l] > 0]
                            )
    @classmethod
    def walk_permutation_tree(cls, counts, on_visit, indices=None, dim=None, num_permutations=None):
        """
        Just a general purpose method that allows us to walk the permutation
        tree built from counts and apply a function every time a node is visited.
        This can be very powerful for building algorithms that need to consider every permutation of
        an object.

        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        # We walk the counts tree.
        # This is adapted directly from the tree approach to
        # building permutations given a set of indices, except here we just loop over all
        # indices.

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        tree_data = np.zeros((dim, 2), dtype=int)
        depth = 0  # where we're currently writing
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        tree_data[depth, 1] = num_permutations
        init_counts = counts
        classes = np.arange(len(counts))
        counts = np.copy(counts)  # we're going to modify this in-place
        nterms = len(counts)

        perm = np.zeros(dim, dtype=int)

        if indices is None:
            indices = range(num_permutations)

        allow_bracktracking = False
        for idx in indices:
            # we back track directly to where the sum of the subtotal and the current num_before
            # is greater than the target index
            if allow_bracktracking:
                tree_sums = tree_data[:depth, 0] + tree_data[1:depth + 1, 1]
                target_depth = np.where(tree_sums > idx)[0]
                if len(target_depth) > 0:
                    target_depth = np.max(target_depth)
                    for d in range(depth - 1, target_depth, -1):
                        j = perm[d]
                        depth -= 1
                        counts[j] += 1
                    tree_data[depth, 0] = tree_data[depth-1, 0]
                else:
                    # means we need to backtrack completely
                    # so why even walk?
                    depth = 0
                    counts = init_counts.copy()
                    tree_data[depth, 1] = num_permutations
                    tree_data[depth, 0] = 0
            else:
                allow_bracktracking = True

            done = False
            for i in range(depth, dim):  # we only need to do at most cur_dim writes
                # We'll get each element 1-by-1 in an O(d) fashion.
                # This isn't blazingly fast but it'll work okay

                # loop over the classes of elements and see at which point dropping an element exceeds the current index
                # which tells is that the _previous_ term was the correct one
                for j in range(nterms):
                    if counts[j] == 0:
                        continue

                    subtotal = cls._subtree_counts(tree_data[depth, 1], dim - depth, counts, j)
                    test = tree_data[depth, 0] + subtotal
                    if test > idx:  # or j == nterms-1: there's got to be _some_ index at which we get past idx I think...
                        depth += 1
                        counts[j] -= 1
                        perm[i] = classes[j]
                        tree_data[depth, 1] = subtotal
                        tree_data[depth, 0] = tree_data[depth - 1, 0]
                        if tree_data[depth, 0] == idx:
                            # we know that every next iteration will _also_ do an insertion
                            # so we can just do that all at once
                            insertion = np.concatenate(
                                [np.full(counts[l], classes[l], dtype=perm.dtype) for l in range(nterms) if counts[l] > 0]
                            )
                            perm[i + 1:] = insertion
                            done = True
                        break
                    else:
                        tree_data[depth, 0] = test

                if done:
                    on_visit(idx, perm, counts, depth, tree_data)
                    break

    @classmethod
    def descend_permutation_tree_indices(cls, perms, on_visit, classes=None, counts=None, dim=None, assume_sorted=False, num_permutations=None):
        """
        Not sure what to call this exactly, but given that `walk_permutation_tree` maps onto `permutations_from_indices`
        this is the counterpart that basically walks _down_ the way `permutation_indices` would.
        I guess this is basically a BFS type approach of something?

        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        raise NotImplementedError("The idea is good but I don't need it yet")

        perms = np.asanyarray(perms)

        if classes is None or counts is None:
            classes, counts = cls.get_permutation_class_counts(perms[0])

        smol = perms.ndim == 1
        if smol:
            assume_sorted = True
            perms = np.reshape(perms, (1, len(perms)))

        if not assume_sorted:
            sorting = np.lexsort(-np.flip(perms, axis=1).T)
            perms = perms[sorting,]
        else:
            sorting = None

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        tree_data = np.zeros((dim, 2), dtype=int)
        cur_dim = dim - 1
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        tree_data[cur_dim, 1] = num_permutations
        ndim = dim
        # we make a constant-time lookup for what a value maps to in
        # terms of position in the counts array
        class_map = {v: i for i, v in enumerate(classes)}
        init_counts = counts
        counts = np.copy(counts)  # we're going to modify this in-place
        nterms = len(counts)
        # stack = collections.deque()  # stack of current data to reuse

        # determine where each successive permutation differs so we can know how much to reuse
        permutation_indices = np.arange(ndim)
        diffs = np.not_equal(perms[:-1], perms[1:])
        for sn, state in enumerate(perms):
            if sn > 0:
                # we reuse as much work as we can by only backtracking where we need to
                agree_pos = np.where(diffs[sn - 1])[0][
                    0]  # where the first disagreement occurs...I'd like this to be less inefficient
                num_diff = ndim - agree_pos  # number of differing states
                if num_diff == 0:  # same state so just reuse the previous value
                    # if inds[sn - 1] == -1:
                    #     raise ValueError("permutation {} tried to reused bad value from permutation {}".format(
                    #         perms[sn], perms[sn - 1]
                    #     ))
                    # inds[sn] = inds[sn - 1]
                    continue
                elif agree_pos == 0:
                    # no need to actually backtrack when we know what we're gonna get
                    cur_dim = ndim - 1
                    tree_data[cur_dim, 1] = num_permutations
                    tree_data[cur_dim, 0] = 0
                    counts = init_counts.copy()
                else:
                    prev = perms[sn - 1]
                    # at this point cur_dim gives us the number of trailing
                    # digits that are equivalent in the previous permutation
                    # so we only need to back-track to where the new state begins to
                    # differ from the old one,
                    for i in range(ndim - cur_dim - 2, agree_pos - 1, -1):
                        j = class_map[prev[i]]
                        counts[j] += 1
                        cur_dim += 1
                    # I'm not sure why this didn't break earlier without this...
                    tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]

                    state = state[agree_pos:]

            # we loop through the elements in the permutation and
            # add up number of elements in the subtree that would precede
            # the state in reverse-lexicographic order
            for i, el in enumerate(state):
                for j in range(nterms):
                    if counts[j] == 0:
                        continue
                    subtotal = cls._subtree_counts(tree_data[cur_dim, 1], cur_dim + 1, counts, j)
                    if classes[j] == el:
                        cur_dim -= 1
                        counts[j] -= 1
                        tree_data[cur_dim, 1] = subtotal
                        permutation_indices[cur_dim] = j
                        tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]
                        break
                    else:
                        tree_data[cur_dim, 0] += subtotal

                # short circuit if we've gotten down to a terminal node where
                # there is just one unique element
                if tree_data[cur_dim, 1] == 1:
                    # now...we know what the permutation looks like in terms of the classes itself????
                    # except...I'm not totally sure how to put this
                    permutation_indices[cur_dim:] = np.arange(cur_dim, )
                    break

            on_visit(tree_data[cur_dim, 0], permutation_indices, tree_data)

class IntegerPartitionPermutations:
    """
    Provides tools for working with permutations of a given integer partition
    """
    def __init__(self, num, dim=None):
        self.int = num
        if dim is None:
            dim = num
            self.partitions = IntegerPartitioner.partitions(num, pad=True)
        else:
            if dim <= num:
                self.partitions = IntegerPartitioner.partitions(num, pad=True, max_len=dim)
            else:
                parts_basic = IntegerPartitioner.partitions(num, pad=True, max_len=num)
                self.partitions = np.concatenate(
                    [
                            parts_basic,
                            np.zeros((len(parts_basic), dim - num), dtype=int)
                        ],
                    axis=1
                )

        self.dim = dim

        self._class_counts = np.asanyarray([ tuple(np.flip(y) for y in np.unique(x, return_counts=True)) for x in self.partitions ], dtype=object)
        self.partition_counts = np.array([UniquePermutations.count_permutations(x[1]) for x in self._class_counts])
        self._cumtotals = np.cumsum(np.concatenate([[0], self.partition_counts[:-1]]), axis=0)
        self._num_terms = np.sum(self.partition_counts)

    @property
    def num_elements(self):
        return self._num_terms

    def get_partition_permutations(self, return_indices=False):
        """


        :return:
        :rtype:
        """

        return [UniquePermutations.get_subsequent_permutations(p,
                                                                return_indices=return_indices,
                                                                classes=c[0], counts=c[1]) for p,c in zip(self.partitions, self._class_counts)]

    def _get_partition_splits(self, perms, split_method='direct'):
        """

        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        partitions = np.flip(np.sort(perms, axis=1), axis=1)
        if split_method == '2D':
            partitions = np.ascontiguousarray(partitions)
            nrows, ncols = partitions.shape
            dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                     'formats': ncols * [partitions.dtype]}
            filter_inds = partitions.view(dtype)
            _, mask = np.unique(filter_inds, axis=0, return_inverse=True)
            sorting = np.argsort(mask)
            # now we use `unique` again to split mask position in the sorted array
            _, inds = np.unique(mask[sorting], return_index=True)
            subu = np.array([p[0] for p in partitions[sorting,]])
            uinds = IntegerPartitioner.partition_indices(subu, sums=np.full(len(subu), self.int))
        else:
            partition_inds = IntegerPartitioner.partition_indices(partitions, sums=np.full(len(perms), self.int))
            uinds, mask = np.unique(partition_inds, return_inverse=True)
            sorting = np.argsort(mask)
            # now we use `unique` again to split mask position in the sorted array
            _, inds = np.unique(mask[sorting], return_index=True)

        return uinds, sorting, inds

    def get_full_equivalence_class_data(self, perms, split_method='direct', return_permutations=False):
        """
        Returns the equivalence class data of the given permutations
        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        # convert perms into their appropriate partitions
        # get the indices of those and then split
        uinds, sorting, inds = self._get_partition_splits(perms, split_method=split_method)

        groups = np.split(perms[sorting,], inds)[1:]
        partition_data = self._class_counts[uinds]

        if return_permutations:
            # now I need to relate perms to the "standard" permutation as a set of
            # swaps, since it is only in this approach that various properties will hold...
            # this means [1, 0, 0, 0, 0] -> [0, 0, 1, 0, 0] by [2, 1, 0, 3, 4] not [1, 2, 0, 3, 4]
            partition_sorting = np.argsort(np.argsort(-perms, axis=1), axis=1)  # these are the "permutations" in IntegerPartitionPermutations
            partition_groups = np.split(partition_sorting[sorting,], inds)[1:]

            # partition_groups = []
            # for group, data in zip(groups, partition_data):
            #     partition_groups.append(
            #         UniquePermutations.get_permutation_swaps(group, classes=data[1], counts=data[2])
            #     )
        else:
            partition_groups = None

        return uinds, partition_data, partition_groups, groups, sorting, self._cumtotals[uinds]

    def get_equivalence_classes(self, perms, split_method='direct', return_permutations=True):
        """
        Returns the equivalence classes and permutations of the given permutations
        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        _, partition_data, partition_groups, _, sorting, totals = self.get_full_equivalence_class_data(perms, split_method=split_method, return_permutations=return_permutations)

        return [(c[0], c[1], p) for c,p in zip(partition_data, partition_groups)], totals, sorting

    def get_partition_permutation_indices(self, perms, split_method='direct'):
        """
        Assumes the perms all add up to the stored int
        They're then grouped by partition index and finally
        Those are indexed

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        uinds, partition_data, partition_groups, groups, sorting, totals = self.get_full_equivalence_class_data(perms, split_method=split_method)

        subinds = [
            s + UniquePermutations.get_permutation_indices(g, d[0], d[1], self.dim)
            for d, g, s in zip(partition_data, groups, totals)
        ]

        # raise Exception(groups, subinds)

        subinds = np.concatenate(subinds, axis=0)
        inv = np.argsort(sorting)

        return subinds[inv]

    def get_partition_permutations_from_indices(self, indices, assume_sorted=False):
        """
        Assumes the perms all add up to the stored int
        They're then grouped by partition index and finally
        Those are indexed

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        smol = isinstance(indices, (int, np.integer))
        if smol:
            assume_sorted = True
            indices = np.array([smol])

        # we sort the indices and then use that to split them into groups
        # by comparing them to the `_cumsums`
        if not assume_sorted:
            sorting = np.argsort(indices)
            indices = indices[sorting]
        else:
            sorting = None

        insertion_spots = np.searchsorted(self._cumtotals-1, indices)
        uinds, inds = np.unique(insertion_spots, return_index=True)
        groups = np.split(indices, inds)[1:]
        uinds = uinds - 1

        shifted_groups = [g - s for g,s in zip(groups, self._cumtotals[uinds])]

        # raise Exception(shifted_groups, self._cumtotals)

        # raise Exception(uinds, shifted_groups)

        partition_data = self._class_counts[uinds]
        perms = [
            UniquePermutations.get_permutations_from_indices(d[0], d[1], g, self.dim)
            for d, g in zip(partition_data, shifted_groups)
        ]

        cats = np.concatenate(perms, axis=0)
        if sorting is not None:
            inv = np.argsort(sorting)
            cats = cats[inv]

        return cats

    def __repr__(self):
        return "{}({}, dim={}, nels={})".format(type(self).__name__, self.int, self.dim, self.num_elements)

class EmptyIntegerPartitionPermutations(IntegerPartitionPermutations):

    def __init__(self, num, dim=None):
        self.int = num
        if dim is None:
            dim = 1

        self.dim = dim

        self._class_counts = np.array([ [np.array([0]), np.array([dim])] ], dtype=object)
        self.partition_counts = np.array([], dtype=int)
        self._cumtotals = np.array([0, 1])
        self._num_terms = 1

    def get_partition_permutation_indices(self, perms, split_method=None):
        """

        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """
        woof = np.asanyarray(perms)
        if woof.ndim == 1:
            return 0
        else:
            return np.zeros(len(woof), dtype='int8')

    def get_partition_permutations_from_indices(self, indices, assume_sorted=None):
        """

        :param indices:
        :type indices:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        if isinstance(indices, (np.integer, )):
            return np.zeros(self.dim, dtype='int8')
        else:
            return np.zeros((len(indices), self.dim), dtype='int8')

    def _get_partition_splits(self, perms, split_method='direct'):
        """

        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        uinds = [0]
        splits = [0]
        sorting = np.arange(len(perms))

        return uinds, splits, sorting

class SymmetricGroupGenerator:
    """
    I don't know what to call this.
    Manages elements of the symmetric group up to arbitrary size.
    Basically just exists to merge all of the prior integer partition/permutation stuff over many integers
    which makes it easier to calculate direct products of terms
    """

    def __init__(self, dim):
        """
        :param dim: the padding length of every term (needed for consistency reasons)
        :type dim: int
        """

        self.dim = dim
        self._partition_permutations = [EmptyIntegerPartitionPermutations(0, dim=self.dim)] #type: list[IntegerPartitionPermutations]
        self._counts = [1] #type: list[int]

    def _get_partition_perms(self, iterable):
        """

        :param iterable:
        :type iterable:
        :return:
        :rtype: tuple[list[IntegerPartitionPermutations], list[int]]
        """

        inds = list(iterable)
        max_n = max(inds)
        min_n = min(inds)
        if min_n < 0:
            raise ValueError("can't deal with partitions for negative integers")

        if max_n >= len(self._partition_permutations):
            new_stuff = [IntegerPartitionPermutations(n, dim=self.dim) for n in range(len(self._partition_permutations), max_n+1)]
            self._partition_permutations = self._partition_permutations + new_stuff #type: list[IntegerPartitionPermutations]
            new_counts = [x.num_elements for x in new_stuff]
            self._counts = self._counts + new_counts
            self._cumtotals = np.concatenate([[0], np.cumsum(self._counts)])

        t1 = [self._partition_permutations[n] for n in inds] #type: list[IntegerPartitionPermutations]
        t2 = [self._cumtotals[n] for n in inds] #type: list[int]

        return t1, t2

    def get_terms(self, n, flatten=True):
        """
        Returns permutations of partitions
        :param n:
        :type n:
        :return:
        :rtype:
        """

        if isinstance(n, int):
            n = [n]

        partitioners, counts = self._get_partition_perms(n)
        perms = [wat.get_partition_permutations() for wat in partitioners]
        if flatten:
            perms = np.concatenate([np.concatenate(x, axis=0) for x in perms], axis=0)
        return perms

    def to_indices(self, perms, sums=None, assume_sorted=False):
        """
        Gets the indices for the given permutations.
        First splits by sum then allows the held integer partitioners to do the rest
        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        perms = np.asanyarray(perms)
        smol = perms.ndim == 1
        if smol:
            assume_sorted = True
            perms = perms[np.newaxis]

        if sums is None:
            sums = np.sum(perms, axis=1)

        if not assume_sorted and len(perms) > 1:
            sorting = np.argsort(sums)
            sums = sums[sorting]
            perms = perms[sorting]
        else:
            sorting = None

        usums, inds = np.unique(sums, return_index=True)
        groups = np.split(perms, inds)[1:]

        partitioners, shifts = self._get_partition_perms(usums)
        indices = np.concatenate([ s + p.get_partition_permutation_indices(g) for p,g,s in zip(partitioners, groups, shifts)], axis=0)

        if sorting is not None:
            indices = indices[np.argsort(sorting)]

        return indices

    def from_indices(self, indices, assume_sorted=False):
        """
        Gets the permutations for the given indices.
        First splits into by which integer partitioner is the generator and lets
        the partitioner do the rest

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        smol = isinstance(indices, (int, np.integer))
        if smol:
            indices = np.array([smol])

        if not assume_sorted:
            sorting = np.argsort(indices)
            indices = indices[sorting]
        else:
            sorting = None

        insertion_spots = np.searchsorted(self._cumtotals - 1, indices)
        uinds, inds = np.unique(insertion_spots, return_index=True)
        groups = np.split(indices, inds)[1:]
        uinds = uinds - 1

        partitioners, shifts = self._get_partition_perms(uinds)

        shifted_groups = [g - s for g,s in zip(groups, shifts)]
        # raise Exception(partitioners, shifted_groups, shifts)
        # wat = partitioners[0] # type: IntegerPartitionPermutations
        perms = np.concatenate([ p.get_partition_permutations_from_indices(g) for p,g in zip(partitioners, shifted_groups)], axis=0)

        if sorting is not None:
            perms = perms[np.argsort(sorting),]

        return perms

    def _build_direct_sums(self, input_perm_classes, counts, classes, return_indices=False, filter_negatives=True):
        """
        Creates direct sums of `input_perm_classes` with the unique permutations of `classes` where
        each of the classes has the same counts (just so we don't have to walk the tree as much)
        The `input_perm_classes` are tuples like `(classes, counts, perms)` where the perms are sorted
        which ensures that every subsequent addition is _also_ sorted

        :param input_perm_classes:
        :type perms:
        :param counts:
        :type counts:
        :param classes:
        :type classes:
        :return:
        :rtype:
        """

        input_perm_classes = [(x[0], x[1], x[2], UniquePermutations.get_standard_permutation(x[1], x[0])) for x in input_perm_classes]

        # set up storage
        num_perms = UniquePermutations.count_permutations(counts)

        perm_counts = [len(x[2]) for x in input_perm_classes]
        cum_counts = np.cumsum([0] + perm_counts)
        total_perm_count = np.sum(perm_counts)
        ndim = len(input_perm_classes[0][2][0])

        storage = np.zeros((total_perm_count, num_perms, len(classes), ndim), dtype=input_perm_classes[0][2][0].dtype)
        if return_indices:
            indices = np.zeros((total_perm_count, num_perms, len(classes)), dtype=int)

        if return_indices and not filter_negatives:
            raise NotImplementedError("Can't return indices without filtering negatives... (as I've currently implemented things)")

        dropped_pairs = [] # set up storage for when things go negative
        def on_visit(idx, perm, cts, depth, tree_data):
            """

            :param idx: 
            :type idx:
            :param perm:
            :type perm:
            :param counts:
            :type counts:
            :param depth:
            :type depth:
            :param tree_data:
            :type tree_data:
            :return:
            :rtype:
            """

            class_perms = np.array([c[perm] for c in classes])

            for i,class_data in enumerate(input_perm_classes):
                cls, cts, perms, rep = class_data
                # we make use of the fact that first adding the `class_perms` and _then_
                # taking all the permutations specified in `perms` will get us to the same place
                # as taking all of the `perms` first and then adding the `class_perms` _if_ we
                # do this for all of the possible perms of `counts` (i.e. if we do this in full)
                # This gives us strict ordering relations that we can make use of and allows us to only calculate
                # the counts once
                new_rep_perm = rep[np.newaxis, :] + class_perms
                negs = None
                if filter_negatives:
                    # if we run into negatives we need to mask them out
                    negs = np.where(new_rep_perm < 0)
                    if len(negs) > 0:
                        negs = negs[0]
                        if len(negs) > 0: # weird numpy shit
                            negs = np.unique(negs[0]) # the j values to drop
                            # now we take the complement
                            dropped_pairs.append((i, idx, negs))

                    comp = np.setdiff1d(np.arange(len(class_perms)), negs)
                    new_perms = new_rep_perm[comp[:, np.newaxis, np.newaxis], perms[np.newaxis, :, :]]
                    # raise Exception(comp, storage[cum_counts[i]:cum_counts[i+1], idx, comp].shape, new_rep_perm[:, perms].shape, new_perms.shape, perms.shape)
                    storage[cum_counts[i]:cum_counts[i+1], idx, comp] = new_perms.transpose(1, 0, 2)
                else:
                    new_perms = new_rep_perm[:, perms].transpose(1, 0, 2)
                    storage[cum_counts[i]:cum_counts[i + 1], idx] = new_perms

                if return_indices:
                    # since we're assured sorting we make use of that when getting indices
                    if negs is None or len(negs) == 0:
                        # print(">>", new_rep_perm)
                        classes_count_data = [UniquePermutations.get_permutation_class_counts(p) for p in new_rep_perm]
                        standard_rep_perms = np.array([UniquePermutations.get_standard_permutation(c[1], c[0]) for c in classes_count_data])
                        padding = self.to_indices(standard_rep_perms)
                        for j in range(len(classes)):
                            indices[cum_counts[i]:cum_counts[i + 1], idx, j] = padding[j] + UniquePermutations.get_permutation_indices(new_perms[j],
                                                                                                                                       classes=classes_count_data[j][0],
                                                                                                                                       counts=classes_count_data[j][1]
                                                                                                                                       )
                    elif len(comp) > 0:
                        classes_count_data = [UniquePermutations.get_permutation_class_counts(p) for p in new_rep_perm[comp]]
                        standard_rep_perms = np.array([UniquePermutations.get_standard_permutation(c[1], c[0]) for c in classes_count_data])
                        # print(standard_rep_perms)
                        padding = self.to_indices(standard_rep_perms)
                        for n,j in enumerate(comp):
                            indices[cum_counts[i]:cum_counts[i + 1], idx, j] = padding[n] + UniquePermutations.get_permutation_indices(new_perms[j],
                                                                                                                                       classes=classes_count_data[n][0],
                                                                                                                                       counts=classes_count_data[n][1]
                                                                                                                                       )

        UniquePermutations.walk_permutation_tree(counts, on_visit)

        if len(dropped_pairs) > 0:
            mask = np.full(storage.shape, True)
            if return_indices:
                ind_mask = np.full(indices.shape, True)
            for pair in dropped_pairs:
                i, idx, negs = pair
                mask[cum_counts[i]:cum_counts[i+1], idx, negs] = False
                if return_indices:
                    ind_mask[cum_counts[i]:cum_counts[i + 1], idx, negs] = False
            storage = storage[mask]
            if return_indices:
                indices = indices[ind_mask]

        storage = storage.reshape((-1, ndim))
        if return_indices:
            indices = indices.flatten()
            return storage, indices
        else:
            return storage

    def take_permutation_rule_direct_sum(self, perms, rules, sums=None,
                                         assume_sorted=False,
                                         return_indices=False):
        """
        Applies `rules` to perms.
        Naively this is just taking every possible permutation of the rules padded to
        get to the appropriate length and then adding that to every element in perms
        and then taking the unique ones.
        We can be more intelligent about how we do this, though, first reducing perms to
        equivalence classes as integer partitions and then making use of that to
        minimize the number of operations we need to do while also ensuring sorting

        :param perms:
        :type perms:
        :param rules:
        :type rules:
        :return:
        :rtype:
        """

        # if dim is None:
        dim = self.dim

        # first up we pad the rules
        rules = [
            np.concatenate([r, [0]*(dim - len(r))]) if len(r) < dim else np.array(r)
            for r in rules
            if len(r) <= dim
        ]

        # get counts so we can split them up
        wat = [UniquePermutations.get_permutation_class_counts(rule, sort_by_counts=True) for rule in rules]
        rule_counts = np.asanyarray(wat, dtype=object)

        # first split by length
        count_lens = np.array([len(x[0]) for x in rule_counts])
        len_sort = np.argsort(count_lens)
        len_invs = np.argsort(len_sort)
        _, len_split = np.unique(count_lens[len_sort], return_index=True)
        rule_counts = rule_counts[len_sort]
        rule_count_splits = np.split(rule_counts, len_split)[1:]
        invs_splits = np.split(len_invs, len_split)[1:]
        # next sort and split the rules for real
        rule_groups = [] # no reason to be fancy here
        rule_inv = []
        # raise Exception(rule_count_splits)
        for split, inv in zip(rule_count_splits, invs_splits):
            rule_counts = np.array([x[1] for x in split], dtype=int)
            split_sort = np.lexsort(np.flip(rule_counts, axis=1).T)
            rule_counts = rule_counts[split_sort,]
            inv = inv[split_sort,]
            split = split[split_sort,]
            ucounts, sub_split = np.unique(rule_counts, axis=0, return_index=True)
            count_splits = np.split(split, sub_split)[1:]

            rule_groups.extend(count_splits)
            rule_inv.append(inv)
        rule_inv = np.concatenate(rule_inv)

        perms = np.asanyarray(perms)
        # next we pad up the perms as needed
        if perms.shape[1] < dim:
            perms = np.concatenate([
                perms,
                np.zeros((perms.shape[0], dim - perms.shape[1]), dtype=perms.dtype)
            ],
                axis=1
            )
        elif perms.shape[1] > dim:
            raise ValueError("with dimension {} can't handle states of dimension {}".format(dim, perms.shape[-1]))

        if sums is None:
            sums = np.sum(perms, axis=1)

        if not assume_sorted:
            sorting = np.argsort(sums)
            sums = sums[sorting]
            perms = perms[sorting]
        else:
            sorting = None

        # next split up the input permutations
        usums, inds = np.unique(sums, return_index=True)
        groups = np.split(perms, inds)[1:]

        partitioners, shifts = self._get_partition_perms(usums)

        class_data = [p.get_equivalence_classes(g) for p,g in zip(partitioners, groups)]

        perm_classes = [c[0] for c in class_data]
        perm_totals = [c[1] for c in class_data]
        perm_inverse = [c[2] for c in class_data]

        # raise Exception(perm_classes, groups)

        perms = []
        if return_indices:
            indices = []

        rule_counts = [group[0][1] for group in rule_groups]
        rule_classes = [[g[0] for g in group] for group in rule_groups]
        for input_classes,base_shift,tots in zip(perm_classes, shifts, perm_totals):
            perm_block = []
            if return_indices:
                ind_block = []
            for counts, classes in zip(rule_counts, rule_classes):
                res = self._build_direct_sums(input_classes, counts, classes,
                                              return_indices=return_indices
                                              )
                if return_indices:
                    perm_block.append(res[0])
                    ind_block.append(res[1])
                else:
                    perm_block.append(res)

            # print(rule_inv, len(perm_block), len(rule_counts), len(rule_classes))
            new_perms = np.concatenate(perm_block, axis=0)
            perms.append(new_perms)
            if return_indices:
                new_inds = np.concatenate(ind_block)
                indices.append(new_inds)


        # now maybe we want to take some kind of inverse????? Or not?????

        perms = np.concatenate(perms, axis=0)
        if return_indices:
            indices = np.concatenate(indices)
            return perms, indices
        else:
            return perms






