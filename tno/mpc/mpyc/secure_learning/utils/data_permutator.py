"""
Contains class used for data permutations
"""
import secrets
from typing import Callable, Optional

import numpy as np
from mpyc.runtime import mpc

from tno.mpc.mpyc.secure_learning.exceptions import SecureLearnValueError
from tno.mpc.mpyc.secure_learning.utils.types import Matrix, SecNumTypesTV, SeqMatrix
from tno.mpc.mpyc.secure_learning.utils.util_matrix_vec import permute_matrix


class SecureDataPermutator:
    """
    Class for performing data permutations.

    :param secure_permutations: If True, perform permutations
        collaboratively using a secure permutation protocol. If False
        perform local permutations based on a shared random seed
    :param seed: Set the random seed. A shared seed can be generated using
        the refresh_seed method
    """

    def __init__(self, secure_permutations: bool, seed: Optional[int] = None) -> None:
        """
        Constructor method.
        """
        self._seed: Optional[int] = seed
        self.permute_data: Callable[[SeqMatrix[SecNumTypesTV]], Matrix[SecNumTypesTV]]
        if secure_permutations:
            self.permute_data = self.secure_data_permutation
        else:
            self.permute_data = self.insecure_data_permutation

    @property
    def seed(self) -> int:
        """
        Seed used for randomness.

        :raise SecureLearnValueError: Seed has not been set
        :return: Seed used for randomness
        """
        if self._seed is None:
            raise SecureLearnValueError("Seed has not been set.")
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """
        Set new seed and re-initiate randomness generator.

        :param seed: Seed for randomness
        """
        self._seed = seed
        np.random.seed(self._seed)

    async def refresh_seed(self) -> None:
        """
        Generate common seed for future permutations.
        """
        seed = await mpc.transfer(secrets.randbelow(2**32))
        self.seed = sum(seed) % 2**32

    @staticmethod
    def secure_data_permutation(
        matrix: SeqMatrix[SecNumTypesTV],
    ) -> Matrix[SecNumTypesTV]:
        """
        Permute the rows of the provided matrix using a secure permutation
        protocol.

        :param matrix: Matrix to be permuted
        :return: Matrix with shuffled rows
        """
        return permute_matrix(matrix)

    def insecure_data_permutation(
        self, matrix: SeqMatrix[SecNumTypesTV]
    ) -> Matrix[SecNumTypesTV]:
        """
        Locally permute the rows of the provided matrix based on a shared
        random seed.

        :param matrix: Matrix to be permuted
        :raise SecureLearnValueError: Seed has not been set
        :return: Matrix with shuffled rows
        """
        if self._seed is None:
            raise SecureLearnValueError("Seed has not been set")
        permutation: Matrix[SecNumTypesTV] = np.random.permutation(
            np.asarray(matrix)
        ).tolist()
        return permutation
