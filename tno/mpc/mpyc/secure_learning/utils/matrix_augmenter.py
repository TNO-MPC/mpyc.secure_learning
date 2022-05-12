"""
Contains the MatrixAugmenter class
"""
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, Tuple, TypeVar, Union, overload

from tno.mpc.mpyc.secure_learning.utils.types import (
    Matrix,
    SeqMatrix,
    Vector,
    seq_to_list,
)

TemplateType = TypeVar("TemplateType")


class MatrixAugmenter(SeqMatrix[TemplateType]):
    """
    Representation of an augmented matrix build from multiple sequence-of-
    sequences matrices.

    Multiple identical-length matrices can be augmented to the MatrixAugmenter
    object. The object stores all matrices in a dictionary so that they can be
    retrieved individually. The MatrixAugmenter itself is a sequence; in
    particular, it behaves as a larger sequence-of-sequences. An update to the
    augmented matrix object translates to an update of all of the individual
    matrices.
    """

    def __init__(self) -> None:
        """
        Constructor method.
        """
        self._matrices: Dict[str, Matrix[TemplateType]] = OrderedDict()

    @overload
    def __getitem__(self, index: int) -> Vector[TemplateType]:
        ...

    @overload
    def __getitem__(self, index: slice) -> "MatrixAugmenter[TemplateType]":
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Vector[TemplateType], "MatrixAugmenter[TemplateType]"]:
        """
        Get a row of the augmented matrix.

        :param index: Row index of the augmented matrix
        :return: Row of the augmented matrix
        """
        if isinstance(index, int):
            return list(
                chain.from_iterable(matrix[index] for matrix in self._matrices.values())
            )
        matrix_augmenter: "MatrixAugmenter[TemplateType]" = MatrixAugmenter()
        matrix_augmenter._matrices = self._matrices.copy()
        for key, val in matrix_augmenter._matrices.items():
            matrix_augmenter._matrices[key] = val[index]
        return matrix_augmenter

    def __len__(self) -> int:
        """
        Length of the augmented matrix.

        :return: Number of rows in the augmented matrix
        """
        return len(self._matrices[next(iter(self._matrices.keys()))])

    def __str__(self) -> str:
        """
        String representation of the augmented matrix.

        States the contents of the augmented matrix, where individual matrices
        are separated by | and specified by their key in a header.

        :return: String representation
        """
        header = " | ".join(self._matrices.keys())
        content = ""
        for augmented_rows in zip(*self._matrices.values()):
            content += "\n" + " | ".join(str(row) for row in augmented_rows)
        return header + content

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Return the shape of the augmented matrix.

        :return: Number of rows and columns
        """
        return len(self), sum(len(matrix[0]) for matrix in self._matrices.values())

    def keys(self) -> Any:
        """
        Provide a set-like object providing a view on the keys of the
        individual matrices.

        :return:  set-like object providing a view on the keys of the
            individual matrices
        """
        return self._matrices.keys()

    @staticmethod
    def _validate_matrix_format(matrix: SeqMatrix[TemplateType]) -> None:
        """
        Validates that the provided object is a sequence of sequences.

        :param matrix: Object to be validated
        :raise TypeError: Object not a sequence of sequences
        """
        if not hasattr(matrix, "__getitem__"):
            raise TypeError(
                f"Expected sequence of sequences as input,\
                    but received {type(matrix)}."
            )
        if not hasattr(matrix[0], "__getitem__"):
            raise TypeError(
                f"Expected sequence of sequences as input, \
                    but received sequence of {type(matrix[0])}."
            )

    def augment(self, key: str, matrix: SeqMatrix[TemplateType]) -> None:
        """
        Augments a new matrix to the existing augmented matrix.

        :param key: Key used for storing and retrieving the new matrix
        :param matrix: Matrix to be augmented
        :raise KeyError: Key already in use
        """
        if key in self._matrices:
            raise KeyError("Key already in use")
        self._update_key(key, matrix)

    def update_key(self, key: str, matrix: SeqMatrix[TemplateType]) -> None:
        """
        Updates an existing matrix in the augmented matrix.

        :param key: Key for retrieving the existing matrix
        :param matrix: New values for matrix
        :raise KeyError: Key not in use
        """
        if key not in self._matrices:
            raise KeyError("Unknown key provided")
        self._update_key(key, matrix)

    def _update_key(self, key: str, matrix: SeqMatrix[TemplateType]) -> None:
        """
        Stores or updates a matrix in the matrix dictionary.

        :param key: Key for storing or retrieving the existing matrix
        :param matrix: New values for matrix
        :raise ValueError: Matrix dimensions inconsistent with existing
            augmented matrix
        """
        self._validate_matrix_format(matrix)
        if self._matrices and len(matrix) != len(self):
            raise ValueError(
                f"Length of provided matrix is inconsistent with \
                    existing matrix. Expected {len(self)} rows, \
                        but received {len(matrix)}."
            )
        self._matrices[key] = seq_to_list(matrix)

    def update(self, matrix: SeqMatrix[TemplateType]) -> None:
        """
        Update augmented matrix in its entirety.

        All individual matrices are updated accordingly.

        :param matrix: New augmented matrix
        :raise ValueError: Number of columns of new augmented matrix does not
            agree with old number of columns
        """
        self._validate_matrix_format(matrix)
        if len(matrix[0]) != self.shape[1]:
            raise ValueError(
                f"Expected matrix of dimensions {self.shape}, \
                    but received dimensions {(len(matrix), len(matrix[0]))}."
            )
        offset = 0
        for key, val in self._matrices.items():
            submat_col_length = len(val[0])
            self._matrices[key] = seq_to_list(
                [row[offset : offset + submat_col_length] for row in matrix]
            )
            offset += submat_col_length

    def retrieve(self, key: str) -> Matrix[TemplateType]:
        """
        Retrieve individual matrix.

        :param key: Key for retrieving the matrix
        :return: Requested matrix
        """
        return self._matrices[key]

    def delete(self, key: str) -> None:
        """
        Delete individual matrix.

        :param key: Key of matrix
        """
        del self._matrices[key]
