"""
Utilities for working with permutations and permutation indexing
"""

import numpy as np
import collections, functools as ft

__all__ = [
    "PartitionPermutationIndexer",
    "IntegerPartitioner",
    "UniquePermutations",
    "IntegerPartitionPermutations"
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
            n = np.asanyarray(n)
            M = np.asanyarray(M)
            l = np.asanyarray(l)

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

        parts = np.asanyarray(parts)
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
                # print(np.column_stack([sums, parts[:, i:], subsums, num_before[inds]]))
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
            skipped = cls.get_permutation_indices(classes, counts, initial_permutation, num_permutations=total_perms)
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

    @classmethod
    def get_permutation_indices(cls, classes, counts, perms, assume_sorted=False, dim=None, num_permutations=None):
        """
        Classmethod interface to get indices for permutations
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

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
        counts = np.copy(counts) # we're going to modify this in-place
        nterms = len(counts)
        # stack = collections.deque()  # stack of current data to reuse

        # determine where each successive permutation differs so we can know how much to reuse
        diffs = np.not_equal(perms[:-1], perms[1:]).astype('byte')

        inds = np.full((len(perms),), -1)
        for sn, state in enumerate(perms):
            if sn > 0:
                # we reuse as much work as we can by only backtracking where we need to
                agree_pos = np.where(diffs[sn-1])[0][0] # where the first disagreement occurs...I'd like this to be less inefficient
                num_diff = ndim - agree_pos  # number of differing states
                if num_diff == 0:  # same state so just reuse the previous value
                    if inds[sn - 1] == -1:
                        raise ValueError("permutation {} tried to reused bad value from permutation {}".format(
                            perms[sn], perms[sn - 1]
                        ))
                    inds[sn] = inds[sn - 1]
                    continue

                prev = perms[sn-1]
                # at this point cur_dim gives us the number of trailing
                # digits that are equivalent in the previous permutation
                # so we only need to back-track to where the new state begins to
                # differ from the old one,
                for i in range(ndim - cur_dim - 2, agree_pos-1, -1):
                    j = class_map[prev[i]]
                    counts[j] += 1
                    tree_data[cur_dim, 1] = 0
                    cur_dim += 1
                # print(ndim-agree_pos)
                # print("<<", cur_dim, tree_data[:, 1], counts)
                state = state[agree_pos:]

            # we loop through the elements in the permutation and
            # add up number of elements in the subtree that would precede
            # the state in reverse-lexicographic order
            for i, el in enumerate(state):
                for j in range(nterms):
                    if counts[j] == 0:
                        continue
                    # print(cur_dim, tree_data[:, 1], counts)
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
                    # print(">>", cur_dim, tree_data[:, 1], counts)
                    break

            inds[sn] = tree_data[cur_dim, 0]

        if sorting is not None:
            inds = inds[np.argsort(sorting)]
        elif smol:
            inds = inds[0]

        return inds

class IntegerPartitionPermutations:
    """
    Provides tools for working with permutations of a given integer partition
    """
    def __init__(self, num, dim=None):
        self.int = num
        if dim is None:
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

        self._class_counts = np.asanyarray([ tuple(np.flip(y) for y in np.unique(x, return_counts=True)) for x in self.partitions ], dtype=object)
        self.partition_counts = np.array([UniquePermutations.count_permutations(x[1]) for x in self._class_counts])
        self._cumtotals = np.cumsum(self.partition_counts)

    def get_partition_permutations(self, return_indices=False):
        """


        :return:
        :rtype:
        """

        return [UniquePermutations.get_subsequent_permutations(p,
                                                                return_indices=return_indices,
                                                                classes=c[0], counts=c[1]) for p,c in zip(self.partitions, self._class_counts)]

    def get_partition_permutation_indices(self, perms, split_method='2D'):
        """
        Assumes the perms all add up to the stored int
        They're then grouped by partition index and finally
        Those are indexed

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        # convert perms into their appropriate partitions
        # get the indices of those and then split
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
            groups = np.split(partitions[sorting,], inds)[1:]
            subu = np.array([p[0] for p in groups])
            uinds = IntegerPartitioner.partition_indices(subu, sums=np.full(len(subu), self.int))
        else:
            partition_inds = IntegerPartitioner.partition_indices(partitions, sums=np.full(len(perms), self.int))
            uinds, mask = np.unique(partition_inds, return_inverse=True)
            sorting = np.argsort(mask)
            # now we use `unique` again to split mask position in the sorted array
            _, inds = np.unique(mask[sorting], return_index=True)
            groups = np.split(partitions[sorting,], inds)


        raise Exception(uinds, groups)

PermutationStateKey = collections.namedtuple("PermutationStateKey", ['non_zero', 'classes'])
class PartitionPermutationIndexer:
    """
    An order statistics tree designed to make it easy to
    get the index of a permutation based on the integer partition
    it comes from, which gives the number of character classes,
    and overall permutation length
    """
    def __init__(self, partition):
        self.ndim = len(partition)
        self.partition = partition
        self.perms = IntegerPartitionPermutations(partition)
        self.non_zero = sum(x for x,c in zip(self.counts, self.classes) if c != 0)
        self.key = PermutationStateKey(self.non_zero, tuple(self.classes))
        self.total_states = self._fac_rat(self.counts)

    @staticmethod
    def _fac_rat(counts):
        import math

        subfac = np.prod([math.factorial(x) for x in counts])
        ndim_fac = math.factorial(np.sum(counts))

        return ndim_fac // subfac

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

    def get_perm_indices(self, states, assume_sorted=True):
        """
        Gets the indices for a set of states.
        Does this by looping through the state, decrementing the appropriate character class,
        computing the number of nodes in the child tree (done based on the initial total states),
        and then adding up all those terms.
        We make use of the assumption that states are sorted to avoid doing more work than necessary
        by reusing stuff from the previous state

        :param state:
        :type state:
        :return:
        :rtype:
        """
        if not assume_sorted:
            raise NotImplementedError("need to sort")

        num_before = 0
        cur_total = self.total_states
        ndim = self.ndim
        cur_dim = self.ndim
        cur_classes = np.copy(self.classes)
        cur_counts = np.copy(self.counts)
        stack = collections.deque() # stack of current data to reuse
        # determine where each successive state differs so we can know how much to reuse
        diffs = np.diff(states, axis=0)
        inds = np.full((len(states),), -1)
        for sn,state in enumerate(states):
            if sn > 0:
                # we reuse as much work as we can by only popping a few elements off of the class/counts stacks
                agree_pos = np.where(diffs[sn-1] != 0)[0][0] # where the first disagreement occurs
                num_diff = ndim - agree_pos # number of differing states
                if num_diff == 0: # same state so just reuse the previous value
                    if inds[sn - 1] == -1:
                        raise ValueError("state {} tried to reused bad value from state {}".format(
                            states[sn], states[sn-1]
                        ))
                    inds[sn] = inds[sn - 1]
                    continue
                # we pop until we know the states agree once more
                # which correc
                stack_depth = len(stack)
                for n in range(stack_depth - agree_pos):
                    # try:
                    num_before, cur_total, cur_classes, cur_counts = stack.pop()
                    # except IndexError:
                    #     raise ValueError("{} doesn't follow {} {} (initial states were not sorted)".format(
                    #         state,
                    #         states[sn-1],
                    #         og_stack,
                    #         num_diff
                    #     ))
                cur_dim = num_diff
                # print("  ::>", "{:>2}".format(sn), len(stack), state)
                state = state[-num_diff:]
                # print("    + ", state)
                # print("    +", "{:>2}".format(num_before))
            # tree traversal, counting leaves in the subtrees
            for i, el in enumerate(state):
                cur_num = num_before
                for j, v in enumerate(cur_classes):
                    subtotal = self._subtree_counts(cur_total, cur_dim, cur_counts, j)
                    if v == el:
                        stack.append((cur_num, cur_total, cur_classes, cur_counts.copy()))
                        cur_total = subtotal
                        cur_dim -= 1
                        cur_counts[j] -= 1
                        if cur_counts[j] <= 0: # just to be safe because why not
                            cur_classes = np.delete(cur_classes, j)
                            cur_counts = np.delete(cur_counts, j)
                        break
                    else:
                        num_before += subtotal
                # short circuit if we've gotten down to a terminal node where
                # there is just one unique element
                if len(cur_classes) == 1:
                    # print("    +", "{:>2}".format(num_before), i, j, cur_total)
                    tup = (cur_num, cur_total, cur_classes, cur_counts)
                    for x in range(len(state) - (i+1)):
                        stack.append(tup)
                    break
            inds[sn] = num_before
            # print("    =", "{:>2}".format(num_before))

        return inds

    def from_perm_indices(self, inds, assume_sorted=False):
        """
        Just loops through the unique permutations
        and returns the appropriate ones for inds.
        Done all in one call for efficiency reasons
        :param inds:
        :type inds: np.ndarray
        :return: permutation array
        :rtype: np.ndarray
        """

        if len(inds) == 0:
            return np.array([], dtype='int8')

        if not assume_sorted:
            sorting = np.argsort(inds)
            inds = inds[sorting]
        else:
            sorting = None

        perms = []
        for n, p in enumerate(IntegerPartitionPermutations(self.partition).get_permutations()):
            while n == inds[0]:
                perms.append(p)
                inds = inds[1:]
                if len(inds) == 0:
                    break
            if len(inds) == 0:
                break
        if len(inds) > 0:
            raise ValueError("indices {} are beyond the number of permutations supported by {}".format(
                inds,
                self
            ))

        perms = np.array(perms, dtype='int8')
        if sorting is not None:
            perms = perms[np.argsort(sorting)]

        return perms

    def __repr__(self):
        return "{}({}, ndim={}, states={})".format(
            type(self).__name__,
            self.partition[np.where(self.partition != 0)],
            self.ndim,
            self.total_states
        )