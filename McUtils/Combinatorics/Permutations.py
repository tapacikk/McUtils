"""
Utilities for working with permutations and permutation indexing
"""

import numpy as np
import collections

__all__ = [
    "PartitionPermutationIndexer",
    "IntegerPartitioner"
]


class IntegerPartitioner:
    def __init__(self, n):
        self.int = n
        self.parts = []

    _partition_counts = None
    @classmethod
    def count_partitions(cls, n, M, l):
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

        if cls._partition_counts is None:
            # we just initialize this 3D array to be the size we need
            cls._partition_counts = np.zeros((n, M, l))

        if n > cls._partition_counts.shape[0]:
            # grow in size along this axis
            # we grow by 2X the amount we really need to
            # so as to amortize the cost of the concatenations (thank you Lyle)
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.zeros_like(cls._partition_counts)
            ],
            axis=0
            )

        if M > cls._partition_counts.shape[1]:
            # double in size along this axis
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.zeros_like(cls._partition_counts)
            ],
            axis=0
            )

        test = cls._partition_counts[n, M, l]
        if test == 0:
            ...
        else:
            return test


    @staticmethod
    def _accell_asc(n, return_len=False):
        """
        Pulled from http://jeromekelleher.net/author/jerome-kelleher.html
        Could easily be translated to C++ if speed is crucial (but this will never be a bottleneck)
        Thought about adding numpy-style optimizations but I don't know that there's
        any reason to since I'm guessing the int addition is more than fast enough
        and numpy overhead would probably just be overhead
        :param n:
        :type n:
        :return:
        :rtype:
        """
        # set up storage for building the partitions
        a = [0] * (n + 1)
        k = 1  # at the end of each run this will be the length of the perm
        y = n - 1
        while k != 0:
            x = a[k - 1] + 1
            k -= 1
            # this could easily be made into a direct for
            # loop which would be good if this were actually compiled
            while 2 * x <= y:
                a[k] = x
                y -= x
                k += 1
            # ditto above
            l = k + 1
            while x <= y:
                a[k] = x
                a[l] = y
                if return_len:
                    yield (k+2, a[:k + 2])
                else:
                    yield a[:k + 2]
                x += 1
                y -= 1
            a[k] = x + y
            y = x + y - 1
            if return_len:
                yield (k, a[:k + 1])
            else:
                yield a[:k + 1]

    @classmethod
    def integer_partitions(cls,
                           n,
                           pad=False,
                           max_len=None,
                           return_len=False
                           ):
        """
        Takes integer partitions and adds extra info for convenience
        :param n:
        :type n:
        :return:
        :rtype:
        """

        if pad:
            n_perms = cls.count_partitions(n)
            if max_len is None:
                storage = np.zeros((n_perms, n))
            else:
                storage = np.zeros((n_perms, max_len))
            if return_len:
                lens = np.zeros((n_perms,))
            else:
                lens = None
            for i,p in enumerate(cls._accell_asc(n, return_len=True)):
                l, p = p
                if max_len is not None and l > max_len:
                    break
                storage[i, :l] = p
                if return_len:
                    lens[i] = l
        else:
            if max_len is None:
                # we can be a little bit more efficient here than otherwise
                # would be possible?
                n_perms = cls.count_partitions(n)
            ...





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
        self.classes, self.counts = np.unique(partition, return_counts=True)
        self.classes = np.flip(self.classes)
        self.counts = np.flip(self.counts)
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

    @staticmethod
    def _unique_permutations(elements):
        """
        From StackOverflow, an efficient enough
        method to get unique permutations
        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        class unique_element:
            def __init__(self, value, occurrences):
                self.value = value
                self.occurrences = occurrences

        def perm_unique_helper(listunique, result_list, d):
            if d < 0:
                yield tuple(result_list)
            else:
                for i in listunique:
                    if i.occurrences > 0:
                        result_list[d] = i.value
                        i.occurrences -= 1
                        for g in perm_unique_helper(listunique, result_list, d - 1):
                            yield g
                        i.occurrences += 1

        if not hasattr(elements, 'count'):
            elements = list(elements)
        eset = set(elements)
        listunique = [unique_element(i, elements.count(i)) for i in eset]
        u = len(elements)
        return list(sorted(list(perm_unique_helper(listunique, [0] * u, u - 1)), reverse=True))

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
        for n, p in enumerate(self._unique_permutations(self.partition)):
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