"""
Types to use for type hinting.
"""
from typing import List, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from mpyc.sectypes import (
    SecureFiniteField,
    SecureFixedPoint,
    SecureInteger,
    SecureObject,
)

SecNumTypes = Union[SecureFixedPoint, SecureInteger, SecureFiniteField]
SecNumTypesTV = TypeVar(
    "SecNumTypesTV", SecureFiniteField, SecureFixedPoint, SecureInteger
)
TemplateType = TypeVar("TemplateType")

Vector = List[TemplateType]
Matrix = List[List[TemplateType]]
SeqVector = Sequence[TemplateType]
SeqMatrix = Sequence[Sequence[TemplateType]]
SecureObjectType = TypeVar("SecureObjectType", bound=SecureObject)
NumpyFloatArray = npt.NDArray[np.float_]
NumpyIntegerArray = npt.NDArray[np.int_]
NumpyObjectArray = npt.NDArray[np.object_]
NumpyNumberArray = Union[NumpyIntegerArray, NumpyFloatArray]
NumpyOrVector = Union[NumpyNumberArray, Vector[float]]
NumpyOrMatrix = Union[NumpyNumberArray, Matrix[float]]


def seq_to_list(matrix: SeqMatrix[TemplateType]) -> Matrix[TemplateType]:
    """
    Convert a sequence-like matrix to a list-like matrix.

    :param matrix: Sequence of sequences.
    :return: List of lists with the same contents.
    """
    return list(list(row) for row in matrix)
