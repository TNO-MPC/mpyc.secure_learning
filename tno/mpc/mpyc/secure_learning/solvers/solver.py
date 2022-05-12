"""
Class for a solver, which can then be used to define
other solvers such as gradient descent or SAG.
"""
import math
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type, Union, overload

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint
from typing_extensions import Literal

from tno.mpc.mpyc.stubs.asyncoro import mpc_coro_ignore, returnType

import tno.mpc.mpyc.secure_learning.utils.util_matrix_vec as mpc_utils
from tno.mpc.mpyc.secure_learning.exceptions import (
    MissingFunctionError,
    SecureLearnUninitializedSolverError,
    SecureLearnValueError,
)
from tno.mpc.mpyc.secure_learning.models.common_gradient_forms import GradientFunction
from tno.mpc.mpyc.secure_learning.regularizers import (
    DifferentiableRegularizer,
    NonDifferentiableRegularizer,
)
from tno.mpc.mpyc.secure_learning.utils import (
    Matrix,
    MatrixAugmenter,
    SecureDataPermutator,
    SeqMatrix,
    Vector,
    seq_to_list,
)


class Solver(ABC):
    """
    Abstract class for a solver, which can then be used to define
    other solvers such as gradient descent or SAG.
    """

    name: str = ""

    def __init__(self) -> None:
        """
        Constructor method.

        Notice that the relevant class variables are instantiated through `init_solver`.
        """
        self.gradient_function: Optional[GradientFunction] = None
        self._list_of_gradient_penalties: List[DifferentiableRegularizer] = []
        self._proximal_function: Optional[NonDifferentiableRegularizer] = None

        self.n: Optional[int] = None
        self.minibatch_size: Optional[int] = None
        self.mu_x: Optional[Vector[SecureFixedPoint]] = None
        self.mu_y: Optional[SecureFixedPoint] = None
        self.yfactor: Optional[SecureFixedPoint] = None
        self.eta0: Optional[Union[float, SecureFixedPoint]] = None
        self.data_permutator: Optional[SecureDataPermutator] = None
        self.permutable_matrix: MatrixAugmenter[SecureFixedPoint] = MatrixAugmenter()
        self.tolerance = 0.0
        self.coef_init: Optional[Vector[SecureFixedPoint]] = None
        self._gradient_function: Optional[GradientFunction] = None
        self.secret_shared_coef_: Optional[Vector[SecureFixedPoint]] = None
        self.nr_epochs: Optional[int] = None
        self.rel_update_diff: Optional[float] = None

    def __str__(self) -> str:
        """
        Returns solver name

        :return: Solver name
        """
        return self.name

    @property
    def nr_inner_iters(self) -> int:
        """
        Return the number of iterations that the inner loop should perform.

        :raise SecureLearnUninitializedSolverError: Occurs when a solver has
            not been fully initialised
        :return: Number of iterations that the inner loop should perform
        """
        if not isinstance(self.n, int):
            raise SecureLearnUninitializedSolverError(
                "Solver has not been fully initialized, \
                    parameter n has not been set."
            )
        if not isinstance(self.minibatch_size, int):
            raise SecureLearnUninitializedSolverError(
                "Solver has not been fully initialized, \
                    parameter minibatch_size has not been set."
            )
        return math.ceil(self.n / self.minibatch_size)

    @staticmethod
    def _initialize_or_verify_initial_coef_(
        coef_init: Optional[Vector[SecureFixedPoint]],
        num_features: int,
        sectype: Type[SecureFixedPoint],
    ) -> Vector[SecureFixedPoint]:
        """
        Parses and verifies initial coefficients vector (possibly
            including intercept).

        Initializes coefficient vector if None was given.
        Verifies that the initial coefficient vector is of the appropriate length.

        :param coef_init: Initial coefficients vector. If None is passed, then
            initialize the coefficient vector as a vector of zeros
        :param num_features: Number of features
        :param sectype: Requested type of initial coefficients vector
        :raise SecureLearnValueError: Provided coefficients vector
            did not pass verification
        :return: Verified initial coefficients vector
        """
        # The intercept that is calculated for non-centered data is returned
        # as the first element of the coefficients vector.
        # Centered data has no intercept.

        n_corr = num_features + 1

        if coef_init is None:
            return [sectype(0) for _ in range(n_corr)]
        if len(coef_init) == n_corr and isinstance(coef_init[0], sectype):
            return coef_init
        raise SecureLearnValueError("Inappropriate initial coefficients vector.")

    def init_solver(
        self,
        total_size: int,
        num_features: int,
        tolerance: float,
        sectype: Type[SecureFixedPoint],
        coef_init: Optional[Vector[SecureFixedPoint]] = None,
        minibatch_size: Optional[int] = None,
        eta0: Optional[float] = None,
    ) -> None:
        """
        Pass configuration to the solver.

        :param total_size: Number of samples in the training data.
        :param num_features: Number of features in the training data.
        :param tolerance: Training stops if the l2 norm of two subsequent coefficient
            vectors is less than the provided tolerance.
        :param sectype: Requested type of initial coefficients vector.
        :param coef_init: Initial coefficients vector. If None is passed, then
            initialize the coefficient vector as a vector of zeros.
        :param minibatch_size: Size of minibatches. Defaults to full batch if
            None is passed.
        :param eta0: Initial learning rate.
        """
        self.tolerance = tolerance
        self.eta0 = eta0
        self.n = total_size

        if minibatch_size is None:
            self.minibatch_size = self.n
        else:
            self.minibatch_size = minibatch_size

        self.coef_init = self._initialize_or_verify_initial_coef_(
            coef_init, num_features=num_features, sectype=sectype
        )

    def set_gradient_function(
        self,
        function: GradientFunction,
    ) -> None:
        """
        Set the gradient function that is used by the solver.

        :param function: Gradient function
        """
        self._gradient_function = function

    @overload
    def _evaluate_gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def _evaluate_gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def _evaluate_gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def _evaluate_gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        """
        Evaluate the gradient function.

        :param X: Independent data
        :param y: Dependent data
        :param coef_: Coefficient vector
        :param grad_per_sample: Return gradient per sample if True, return
            aggregated gradient of all data if False
        :raise MissingFunctionError: No gradient function was initialized
        :return: Value(s) of gradient evaluated with the provided parameters
        """
        if self._gradient_function is None:
            raise MissingFunctionError("Gradient function has not been initialized.")
        return self._gradient_function(X, y, coef_, grad_per_sample=grad_per_sample)

    @overload
    def evaluate_gradient_function_for_minibatch(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        nr_samples_total: int,
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def evaluate_gradient_function_for_minibatch(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        nr_samples_total: int,
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def evaluate_gradient_function_for_minibatch(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        nr_samples_total: int,
        grad_per_sample: bool = ...,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def evaluate_gradient_function_for_minibatch(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        nr_samples_total: int,
        grad_per_sample: bool = False,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        """
        Evaluate the gradient function.

        :param X: Independent data
        :param y: Dependent data
        :param coef_: Coefficient vector
        :param nr_samples_total: Number of samples
        :param grad_per_sample: Return gradient per sample if True, return
            aggregated gradient of all data if False
        :raise MissingGradientFunctionError: No gradient function was
            initialized
        :return: Value(s) of gradient evaluated with the provided parameters
        """
        return mpc_utils.scale_vector_or_matrix(
            len(X) / nr_samples_total,
            self._evaluate_gradient_function(
                X, y, coef_, grad_per_sample=grad_per_sample
            ),
        )

    def add_gradient_penalty_function(
        self, function: DifferentiableRegularizer
    ) -> None:
        """
        Add gradient penalty function to the list of gradient penalty
        functions.

        :param function: Function that evaluates the gradient penalty function in
            a given point
        """
        self._list_of_gradient_penalties.append(function)

    def compute_aggregated_differentiable_regularizer_penalty(
        self,
        coef_: Vector[SecureFixedPoint],
        nr_samples_minibatch: int,
        nr_samples_total: int,
    ) -> Vector[SecureFixedPoint]:
        """
        Compute the aggregated penalty from all gradient penalty functions
        evaluated for the provided gradient. The penalty is weighted by the
        ratio of samples that were used for computing the provided gradient
        over the number of samples in the complete training data.

        :param coef_: Unpenalized objective gradient vector
        :param nr_samples_minibatch: Number of samples that were used for
            computing the given gradient
        :param nr_samples_minibatch: Total number of samples in training data
        :param nr_samples_total: Number of samples
        :raise MissingGradientFunctionError: No gradient function was
            initialized
        :return: Penalized objective gradient vector
        """
        stype = type(coef_[0])
        coef_0 = coef_.copy()

        coef_0[0] = stype(0, integral=False)

        aggregated_penalty = [stype(0, integral=False)] * len(coef_)
        for penalty_func in self._list_of_gradient_penalties:
            aggregated_penalty = mpc.vector_add(
                aggregated_penalty,
                mpc_utils.scale_vector_or_matrix(
                    nr_samples_minibatch / nr_samples_total, penalty_func(coef_0)
                ),
            )
        return aggregated_penalty

    def set_proximal_function(
        self,
        func: NonDifferentiableRegularizer,
    ) -> None:
        """
        Set the proximal function that is used by the solver.

        :param func: A proximal function
        """
        self._proximal_function = func

    @property
    def has_proximal_function(self) -> bool:
        """
        Indicate whether the solver has a proximal function initialized.

        :return: True if the proximal function has been initialized,
            False otherwise
        """
        return self._proximal_function is not None

    def evaluate_proximal_function(
        self,
        coef_: Vector[SecureFixedPoint],
        eta: Union[float, SecureFixedPoint],
    ) -> Vector[SecureFixedPoint]:
        """
        Evaluate the proximal function.

        :param coef_: Coefficient vector
        :param eta: Learning rate
        :raise MissingFunctionError: No proximal function was initialized
        :return: Value of proximal function evaluated with the provided
            parameters
        """
        stype = type(coef_[0])
        coef_0 = coef_.copy()

        coef_0[0] = stype(0, integral=False)
        if self._proximal_function is None:
            raise MissingFunctionError("Proximal function has not been initialized.")
        proximal_result = self._proximal_function(coef_0, eta)
        proximal_result[0] = coef_[0]
        return proximal_result

    @abstractmethod
    def preprocessing(
        self,
        X_init: Matrix[SecureFixedPoint],
        y_init: Vector[SecureFixedPoint],
    ) -> Tuple[Matrix[SecureFixedPoint], Vector[SecureFixedPoint]]:
        """
        Preprocess obtained data.

        May include centering and scaling.

        :param X_init: Independent data
        :param y_init: Dependent data
        :return: Preprocessed independent and dependent data
        """

    @abstractmethod
    def inner_loop_calculation(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_old: Vector[SecureFixedPoint],
        epoch: int,
    ) -> Vector[SecureFixedPoint]:
        """
        Performs one inner-loop iteration for the solver. Inner-loop refers
        to iteratively looping through the data in batches rather than looping
        over the complete data multiple times.

        :param X: Independent data
        :param y: Dependent data
        :param coef_old: Current iterative solution
        :param epoch: Number of times that the outer loop has completed
        :return: Updated iterative solution
        """

    @staticmethod
    def postprocessing(
        coef_predict: Vector[SecureFixedPoint],
    ) -> Vector[SecureFixedPoint]:
        """
        Postprocess the predicted coefficients.

        :param coef_predict: Predicted coefficient vector
        :return: Postprocessed coefficient vector
        """
        return coef_predict

    def iterative_data_permutation(
        self,
        matrix: SeqMatrix[SecureFixedPoint],
    ) -> Matrix[SecureFixedPoint]:
        """
        Permutes a matrix containing both the independent and dependent
        variables $X$ and $y$, respectively.

        :param matrix: The matrix as mentioned in the above summary
        :return: Permuted matrix
        :raise ValueError: when the data permutator is not set yet before this function is called
        """
        if self.nr_inner_iters == 1:
            # Full batch -> Permutation makes no sense
            return seq_to_list(matrix)
        # Permute X and y
        if self.data_permutator is None:
            raise ValueError(
                "The data permutator needs to be set before it can be used"
            )
        return self.data_permutator.permute_data(matrix)

    ###
    # Model training with gradient descent
    ###

    @mpc_coro_ignore
    async def _get_coefficients(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        n_maxiter: int,
        print_progress: bool,
        secure_permutations: bool,
    ) -> Vector[SecureFixedPoint]:
        """
        Compute the model coefficients.
        Only solver-independent calculations are explicitly defined (and called "outer loop").

        :param X: Training data
        :param y: Target vector
        :param n_maxiter: Maximum number of iterations before method stops and result is returned
        :param print_progress: Print progress (epoch number) to standard output
        :param secure_permutations: Perform matrix permutation securely
        :return: vector with (secret-shared) coefficients computed by the solver
        """
        assert n_maxiter > 0 and self.coef_init is not None
        stype = type(self.coef_init[0])
        await returnType(stype, len(self.coef_init))

        X = X.copy()
        y = y.copy()

        ###
        # Solver-specific pre-processing
        ###
        X, y = self.preprocessing(X, y)

        # Initialize data permutator
        self.data_permutator = SecureDataPermutator(
            secure_permutations=secure_permutations
        )

        self.permutable_matrix.augment("X", X)
        self.permutable_matrix.augment(
            "y", mpc_utils.vector_to_matrix(y, transpose=True)
        )

        # Initialize coefficients
        coef_old = self.coef_init.copy()
        coef_new = coef_old.copy()
        coef_oldest = coef_new.copy()

        ###
        # Gradient descent outer loop (solver-independent)
        ###
        for epoch in range(1, n_maxiter + 1):
            if print_progress and epoch % 10 == 1:
                print(f"Epoch {epoch}...")

            # Permutation
            if epoch == 0:
                await self.data_permutator.refresh_seed()

            self.permutable_matrix.update(
                self.iterative_data_permutation(self.permutable_matrix)
            )
            X = self.permutable_matrix.retrieve("X")
            y = mpc_utils.mat_to_vec(
                self.permutable_matrix.retrieve("y"), transpose=True
            )

            ###
            # Gradient descent inner loop (solver-dependent)
            ###
            coef_new = self.inner_loop_calculation(X, y, coef_old, epoch=epoch)
            coef_old = coef_new.copy()

            # Check for convergence
            # Note that (update_diff <= self.tolerance) is True <=>
            # mpc.is_zero_public(update_diff >= self.tolerance) is True
            update_diff = mpc.in_prod(
                mpc.vector_sub(coef_new, coef_oldest),
                mpc.vector_sub(coef_new, coef_oldest),
            )
            prev_norm = mpc.in_prod(coef_oldest, coef_oldest)
            has_converged = mpc.is_zero_public(
                update_diff >= self.tolerance * prev_norm
            )
            if await has_converged:
                break
            coef_oldest = coef_new.copy()
        else:
            warnings.warn(
                "ConvergenceWarning: The maximum number of iterations was reached which means the coef_ did not converge. Standardizing input data, adjusting step-size, lowering tolerance, or increasing the number of iterations may help."
            )

        ###
        # Solver-specific post-processing
        ###
        coef_predict = self.postprocessing(coef_new)

        # Metadata
        self.secret_shared_coef_ = coef_predict
        self.nr_epochs = epoch
        plain_update_diff = float(await mpc.output(update_diff))
        plain_prev_norm = float(await mpc.output(prev_norm))
        try:
            self.rel_update_diff = plain_update_diff / plain_prev_norm
        except ZeroDivisionError:
            warnings.warn(
                "Update difference division by zero, indicating that the \
                    coefficients vector is very small."
            )
            self.rel_update_diff = 0
        return coef_predict
