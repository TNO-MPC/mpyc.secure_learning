"""
The generate method generates a set of unique indicator vectors. It internally
chooses between a direct and an indirect method, where the direct method is
efficient for small sets and the indirect method is efficient for large sets.
A demo of the different methods is presented by running the file as a script.

Usage (Python 3.6):

.. code-block:: python

    >>> from mpyc.runtime import mpc
    >>> import random_unit_vectors
    >>>
    >>> masked_vectors = random_unit_vectors.generate(5,2)
    >>> unmasked_vectors = list(map(lambda x: mpc.run(mpc.output(x)), masked_vectors))
    >>> print(unmasked_vectors)
    [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]
"""
from typing import Any, Awaitable, Coroutine, List, Optional, Sequence, Type

import mpyc.random
from mpyc.runtime import mpc

from tno.mpc.mpyc.stubs.asyncoro import mpc_coro_ignore, returnType

from tno.mpc.mpyc.secure_learning.utils.types import (
    Matrix,
    SecNumTypesTV,
    SeqMatrix,
    Vector,
    seq_to_list,
)
from tno.mpc.mpyc.secure_learning.utils.util_matrix_vec import matrix_sum


async def random_unit_vectors_are_unique(
    p: List[Vector[SecNumTypesTV]],
) -> Coroutine[Any, Any, Awaitable[bool]]:
    """
    Verify whether all passed unit vectors are unique.

    :param p: List of indicator vectors
    :return: True if all indicator vectors are unique, False otherwise
    """
    q = matrix_sum(p)
    return mpc.is_zero_public(mpc.in_prod(q, q) - len(p))


@mpc_coro_ignore
async def _unique_random_unit_vectors(
    sectype: Type[SecNumTypesTV],
    n: int,
    b: int,
    maxcounter: int = 30,
    counter: int = 1,
) -> List[Vector[SecNumTypesTV]]:
    """
    Generate set of random unit vectors.

    :param sectype: Secure type of unit vector elements
    :param n: Length of unit vector
    :param b: Number of unit vectors
    :param maxcounter: Maximum number of tries to generate unique vectors
        before calling indirect method
    :param counter: The counter
    :return: Set of random unit vectors
    """
    await returnType(sectype, b, n)

    p = [mpyc.random.random_unit_vector(sectype, n) for _ in range(b)]

    if await random_unit_vectors_are_unique(p):
        return p
    if counter < maxcounter:
        vectors: List[Vector[SecNumTypesTV]] = await _unique_random_unit_vectors(
            sectype=sectype, n=n, b=b, maxcounter=maxcounter, counter=counter + 1
        )
        return vectors
    print(
        "random_unit_vectors.generate_unique_unit_vectors_direct: \
            max number of recursions reached, now calling \
                generate_unique_unit_vectors_indirect"
    )
    return generate_unique_unit_vectors_indirect(sectype=sectype, n=n, b=b)


def generate_unique_unit_vectors_direct(
    sectype: Type[SecNumTypesTV], n: int, b: int = 1
) -> List[Vector[SecNumTypesTV]]:
    """
    Direct approach for generating random indicator vectors. In this approach,
    we generate sets of random indicator vectors until all vectors are unique.

    Remark: if this method is used often and b is fixed, then the number of
    retries leaks statistical information about the total number of samples.
    Use the indirect method if this is an issue.

    :param sectype: Secure type of unit vector elements
    :param n: Length of unit vector
    :param b: Number of unit vectors
    :return: Set of random unit vectors
    """
    vectors: List[Vector[SecNumTypesTV]] = _unique_random_unit_vectors(sectype, n, b)
    return vectors


# Indirect method
def _random_binary_permutation(
    sectype: Type[SecNumTypesTV], n: int, b: int
) -> Vector[SecNumTypesTV]:
    """
    Returns vector of random ones and zeros.

    :param sectype: Secure type of unit vector elements
    :param n: Length of vector
    :param b: Number of ones in vector
    :return: Vector with b randomly placed ones and n-b zeros
    """
    p = [sectype(1)] * b + [sectype(0)] * (n - b)
    permutation: Vector[SecNumTypesTV] = mpyc.random.shuffle(sectype, p)  # type: ignore[no-untyped-call]
    return permutation


def decompose_random_binary_permutation(
    s: Vector[SecNumTypesTV], b: int
) -> List[Vector[SecNumTypesTV]]:
    """
    Extract a set of indicator vectors from a vector with ones and zeroes.
    Every indicator vector denotes the position of a one in the input vector.

    Assume that the input vector contains exactly 1 one. We extract an
    indicator vector from this vector 's' as follows:

    .. code-block:: python

        >>> eps = 0
        >>> delta = [0]*n
        >>> for i in range(n):
        >>>\tdelta[i] = (1-eps)*s[i]
        >>>\t.delta = eps + delta[i]

    The first line in the for-loop copies the i-th element of `s` as long as
    `eps` equals zero. This remains true until the loop reaches the index `i*`
    such that `s[i] = 1`. Now, `delta[i*]` is set to one and subsequently
    `eps` is set to one. As a consequence, all elements of `delta` with index
    at least `i*` are set to zero. We have thus obtained a indicator vector
    `delta` that has 1-bit on the index that corresponds to the first 1-bit of
    `s`.

    The above approach can be generalized as to obtain `b` indicator  vectors
    from a permutation of `b` 1-bits. Note that all these indicator vectors
    are unique and, by construction, the indicator vectors are sorted
    according to the index of their 1-bit.

    :param s: Vector with zeros and ones
    :param b: Number of ones in s
    :return: Unique set of random indicator vectors
    """
    stype = type(s[0])
    n = len(s)

    eps = [stype(1)] + [stype(0)] * b
    d = [[stype(0)] * n for __ in range(b)]

    for i in range(n):
        for j in range(b):
            d[j][i] = eps[j] * (1 - eps[j + 1]) * s[i]
        for j in range(b):
            eps[j + 1] = eps[j + 1] + d[j][i]

    return d


def generate_unique_unit_vectors_indirect(
    sectype: Type[SecNumTypesTV], n: int, b: int = 1
) -> List[Vector[SecNumTypesTV]]:
    """
    Generate a random binary permutation and then extract unique random
    indicator vectors from the given permutation.

    :param sectype: Secure type of unit vector elements
    :param n: Length of unit vector
    :param b: Number of unit vectors
    :return: Unique set of random indicator vectors
    """
    s = _random_binary_permutation(sectype, n, b)
    return decompose_random_binary_permutation(s, b)


def generate_unique_unit_vectors(
    sectype: Type[SecNumTypesTV], n: int, b: int = 1
) -> List[Vector[SecNumTypesTV]]:
    """
    Generate a set of unique indicator vectors.

    It internally chooses between a direct and an indirect method, where the
    direct method is efficient for small sets and the indirect method is
    efficient for large sets.

    :param sectype: Secure type of unit vector elements
    :param n: Length of unit vector
    :param b: Number of unit vectors
    :raise ValueError: Occurs when 0 <= b <= n when generating indicator
        vectors
    :return: Unique set of random indicator vectors
    """
    if b < 0 or b > n:
        raise ValueError(
            "Ensure that 0 <= b <= n when generating indicator \
                vectors."
        )
    # elif b < 2.1*math.sqrt(n):
    #     return generate_unique_unit_vectors_direct(n, b)  # If b < c*math.sqrt(n), then (asymptotically) the expected number of recursions
    #                                   # is at most exp(c).
    #                                   # c = 2 has been determined experimentally (test2) in case of a single party
    # else:
    #     return generate_unique_unit_vectors_indirect(n, b)

    # In the multi-party setting, generate_unique_unit_vectors_indirect is several orders of magnitute slower
    # (requires too much communication).
    return generate_unique_unit_vectors_direct(sectype, n, b)


def random_matrix_permutation(
    sectype: Type[SecNumTypesTV], row_length: int, rows: Optional[int] = None
) -> List[Vector[SecNumTypesTV]]:
    """
    Return a random permutation matrix.

    :param sectype: Secure type of unit vector elements
    :param row_length: (Row) length of desired matrix
    :param rows: Number of rows to return
    :return: Random permutation matrix
    """
    if rows is None:
        rows = row_length

    matrix = [[sectype(0)] * row_length for __ in range(row_length)]
    for _ in range(row_length):
        matrix[_][_] = sectype(1)
    return permute_matrix(matrix)[:rows]


def permute_matrix(matrix: SeqMatrix[SecNumTypesTV]) -> Matrix[SecNumTypesTV]:
    """
    Permute matrix randomly.

    :param matrix: Matrix to be permuted
    :raise TypeError: Input is not a matrix
    :return: Permutated matrix
    """
    if not isinstance(matrix[0], Sequence):
        raise TypeError("Input is not a matrix.")

    matrix = seq_to_list(matrix)

    stype = type(matrix[0][0])
    rows = len(matrix)
    # Knuth shuffling on vectors
    for row in range(rows - 1):  # Knuth shuffling
        y_r = mpyc.random.random_unit_vector(stype, rows - row)
        X_r = mpc.matrix_prod([y_r], matrix[row:])[0]
        # X_r = matr_sum((mpc_utils.mult_scalar_mul(y_r, X[row:], tr=True))
        d_r = mpc.matrix_prod([[v] for v in y_r], [mpc.vector_sub(matrix[row], X_r)])
        matrix[row] = X_r
        for _ in range(rows - row):
            matrix[row + _] = mpc.vector_add(matrix[row + _], d_r[_])
    return matrix
