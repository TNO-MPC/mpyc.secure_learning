"""
Contains utils for matrices and vectors such as transposing and
secure signing
"""
from typing import Sequence, TypeVar, Union, cast, overload

import mpyc.random
from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint, SecureObject

from tno.mpc.mpyc.stubs.asyncoro import mpc_coro_ignore, returnType

from tno.mpc.mpyc.secure_learning.utils.types import (
    Matrix,
    SecNumTypesTV,
    SeqMatrix,
    Vector,
    seq_to_list,
)

AnyTV = TypeVar("AnyTV")


@overload
def matrix_transpose(matrix: Matrix[SecNumTypesTV]) -> Matrix[SecNumTypesTV]:
    ...


@overload
def matrix_transpose(matrix: Matrix[float]) -> Matrix[float]:
    ...


def matrix_transpose(matrix: Matrix[AnyTV]) -> Matrix[AnyTV]:
    """
    Transpose a list of lists.

    .. code-block:: python

        A = [[31, 64], [32, 68], [33, 72], [34, 76]]
        matrix_transpose(A) == [[31, 32, 33, 34], [64, 68, 72, 76]]

    :param matrix: Matrix stored as list of lists
    :return: Transpose of $A$
    """
    return list(map(list, zip(*matrix)))


@overload
def vector_to_matrix(
    vector: Vector[SecNumTypesTV], transpose: bool = False
) -> Matrix[SecNumTypesTV]:
    ...


@overload
def vector_to_matrix(vector: Vector[float], transpose: bool = False) -> Matrix[float]:
    ...


def vector_to_matrix(vector: Vector[AnyTV], transpose: bool = False) -> Matrix[AnyTV]:
    """
    Convert vector to matrix.

    .. code-block:: python

        vec_to_mat([1, 2, 3]) == [[1, 2, 3]]
        vec_to_mat([1, 2, 3], tr=True) == [[1], [2], [3]]

    :param vector: Row vector to be converted
    :param transpose: Interpret vector as column vector
    :return: Matrix that encapsulates vector
    """
    if not transpose:
        return [vector]
    return [[_] for _ in vector]


def mat_to_vec(matrix: Matrix[AnyTV], transpose: bool = False) -> Vector[AnyTV]:
    """
    Transforms a vector in matrix format to vector format.

    .. code-block:: python

        A = [[1], [2], [3]]
        mat_to_vec(A) == [1, 2, 3]

    :param matrix: Vector in matrix format
    :param transpose: Interpret vector as column vector
    :return: Vector
    """
    if not transpose:
        return matrix[0]
    return [x for [x] in matrix]


def mat_vec_mult(
    matrix: Matrix[SecNumTypesTV],
    vector: Union[Vector[SecNumTypesTV], Vector[float]],
    transpose: bool = False,
) -> Vector[SecNumTypesTV]:
    r"""
    Compute matrix-vector multiplication.

    :param matrix: Matrix input with dimensions $m * r$.
        Dimensions may be $r * m$ when combined with `tr=True`
    :param vector: Vector input of length $r$, treated as a column vector
    :param transpose: If `True`, first transpose `mat`
    :return: Row vector with matrix-vector products
    """
    if transpose:
        matrix = matrix_transpose(matrix)
    else:
        matrix = matrix.copy()

    product = mpc.matrix_prod(matrix, vector_to_matrix(vector, transpose=True))
    return mat_to_vec(product, transpose=True)


@overload
def scale_vector_or_matrix(
    factor: float, x: Vector[SecNumTypesTV]
) -> Vector[SecNumTypesTV]:
    ...


@overload
def scale_vector_or_matrix(
    factor: float, x: Matrix[SecNumTypesTV]
) -> Matrix[SecNumTypesTV]:
    ...


def scale_vector_or_matrix(factor: float, x: Vector[AnyTV]) -> Vector[AnyTV]:
    """
    Corrects a vector or matrix by a given factor.

    :param factor: Factor to scale matrix or vector
    :param x: Vector or matrix to be scaled
    :return: Scaled vector or matrix
    """
    if isinstance(x[0], list):
        return cast(Vector[AnyTV], [scale_vector_or_matrix(factor, _) for _ in x])
    return [factor * _ for _ in x]


def permute_matrix(matrix: SeqMatrix[SecNumTypesTV]) -> Matrix[SecNumTypesTV]:
    """
    Permute matrix randomly.

    :param matrix: Matrix to be permuted
    :raise TypeError: Input is not a matrix
    :return: Permuted matrix
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
        d_r = mpc.matrix_prod([[v] for v in y_r], [mpc.vector_sub(matrix[row], X_r)])
        matrix[row] = X_r
        for _ in range(rows - row):
            matrix[row + _] = mpc.vector_add(matrix[row + _], d_r[_])
    return matrix


@mpc_coro_ignore
# flake8: noqa: C901
async def mult_scalar_mul(
    scalars: Union[float, Vector[float], SecureFixedPoint, Vector[SecureFixedPoint]],
    matrix: Matrix[SecureFixedPoint],
    transpose: bool = False,
) -> Matrix[SecureFixedPoint]:
    """
    Vectorized version of mpc.scalar_mul.

    .. code-block:: python

        scalars = [2, -1]
        mat = [[1, 2], [3, 4], [5, 6]]
        mult_scalar_mul(scalars, mat) == [[2, -2], [6, -4], [10, -6]]

    :param scalars: Vector of scalars
    :param matrix: Matrix of which the columns need to be scaled.
    :param transpose: If `True`, scale the rows of matrix instead.
    :return: Matrix with scaled columns
    """
    matrix = [row[:] for row in matrix]
    rows = len(matrix)
    columns = len(matrix[0])
    scalars_list: Union[Vector[float], Vector[SecureFixedPoint]]
    if not isinstance(scalars, list):
        scalars_list = [scalars] * columns if not transpose else [scalars] * rows  # type: ignore
    else:
        scalars_list = scalars

    stype = type(matrix[0][0])

    frac_length = stype.frac_length
    if not frac_length:
        await returnType(stype, rows, columns)
    else:
        a_integral_first_requirement = isinstance(scalars_list[0], int)
        a_integral_second_requirement = False
        if isinstance(scalars_list[0], SecureFixedPoint):
            a_integral_second_requirement = scalars_list[0].integral
        a_integral = a_integral_first_requirement or a_integral_second_requirement
        await returnType((stype, a_integral and matrix[0][0].integral), rows, columns)

    if not isinstance(scalars_list[0], SecureObject):
        for row in range(rows):
            for column in range(columns):
                matrix[row][column] = matrix[row][column] * (
                    scalars_list[column] if not transpose else scalars_list[row]
                )
    else:
        scalars_list_sec, matrix = await mpc.gather(scalars_list, matrix)
        if frac_length and a_integral:
            for index, row in enumerate(scalars_list_sec):
                scalars_list_sec[index] = row >> frac_length  # NB: no in-place rshift!
        for row in range(rows):
            for column in range(columns):
                matrix[row][column] = matrix[row][column] * (
                    scalars_list_sec[column] if not transpose else scalars_list_sec[row]
                )
            matrix[row] = await mpc._reshare(matrix[row])
        if frac_length and not a_integral:
            for row in range(rows):
                matrix[row] = mpc.trunc(matrix[row], f=frac_length, l=stype.bit_length)
            matrix = await mpc.gather(matrix)
    return matrix


@mpc_coro_ignore
async def matrix_sum(
    matrix: Matrix[SecureFixedPoint], cols: bool = False
) -> Vector[SecureFixedPoint]:
    """
    Securely add all rows in X.

    :param matrix: Matrix to be summed
    :param cols: If `True`, sum the columns of X instead.
    :return: Vector of sums
    """
    matrix = [row[:] for row in matrix]
    if not cols:
        matrix = matrix_transpose(matrix)
    rows = len(matrix)
    stype = type(matrix[0][0])
    frac_length = stype.frac_length
    if not frac_length:
        await returnType(stype, rows)
    else:
        await returnType((stype, matrix[0][0].integral), rows)
    return [mpc.sum(a) for a in matrix]
