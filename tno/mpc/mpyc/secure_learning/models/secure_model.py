"""
Abstract class for secure-learning models.
"""
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from time import time
from typing import Any, List, Optional, Tuple, Union, overload

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint, SecureNumber
from sklearn.model_selection import KFold
from typing_extensions import Literal

from tno.mpc.mpyc.secure_learning import regularizers
from tno.mpc.mpyc.secure_learning.exceptions import (
    SecureLearnTypeError,
    SecureLearnUninitializedSolverError,
    UnknownPenaltyError,
)
from tno.mpc.mpyc.secure_learning.solvers import GD
from tno.mpc.mpyc.secure_learning.solvers.solver import Solver
from tno.mpc.mpyc.secure_learning.utils import Matrix, MatrixAugmenter, Vector

DEFAULT_L1_PENALTY = 1
DEFAULT_L2_PENALTY = 1


class SolverTypes(Enum):
    """
    The possible solver types associated to models.
    """

    GD = auto()


class PenaltyTypes(Enum):
    """
    The possible penalty types associated to models.
    """

    NONE = auto()
    L1 = auto()
    L2 = auto()
    ELASTICNET = auto()


class Model(ABC):
    """
    Abstract secure-learn model class.
    """

    name = ""

    def __init__(
        self,
        solver_type: SolverTypes = SolverTypes.GD,
        penalty: PenaltyTypes = PenaltyTypes.NONE,
        **penalty_args: float,
    ):
        """
        Constructor method.

        :param solver_type: Solver type to use (e.g. Gradient Descent aka GD)
        :param penalty: penalty function (none, l1, l2, or elasticnet)
        :param penalty_args: the coefficient(s) of the given penalty
        """
        self._solver: Optional[Solver] = None
        self.initialize_solver(solver_type, penalty, **penalty_args)

    def __str__(self) -> str:
        """
        String representation of model

        :return: human readable name of the model
        """
        return self.name

    @property
    def solver(self) -> Solver:
        """
        Return solver used by current model.

        :raise SecureLearnUninitializedSolverError: raised when solver is not yet initiated
        :return: Solver used by current model.
        """
        if self._solver is None:
            raise SecureLearnUninitializedSolverError("Solver not initiated.")
        return self._solver

    def initialize_solver(
        self, solver_type: SolverTypes, penalty: PenaltyTypes, **penalty_args: float
    ) -> None:
        """
        Initialize solver.

        :param solver_type: Type of the requested solver.
        :param penalty: Type of penalties
        :param penalty_args: Coefficient(s) of the given penalty
        """
        self._pick_solver(solver_type)
        self._pass_gradient_function_to_solver()
        self._add_penalties(penalty, **penalty_args)

    def _pick_solver(self, solver_type: SolverTypes) -> None:
        """
        Initialize solver of correct type.

        :param solver_type: Type of the requested solver.
        """
        solver: Solver
        if solver_type == SolverTypes.GD:
            solver = GD()
        self._solver = solver

    def _pass_gradient_function_to_solver(self) -> None:
        """
        Initializes solver with gradient function of the model.
        """
        self.solver.set_gradient_function(self.gradient_function)

    def _add_penalties(self, penalty: PenaltyTypes, **penalty_args: float) -> None:
        """
        Sets the given penalty function(s) with the given coefficient(s).

        :args penalty: the type of penalty given (none, l1, l2, or elasticnet)
        :args penalty_args: the coefficient(s) of the given penalty type(s)
        :raise UnknownPenaltyError: if an unknown penalty type is given
        """
        if penalty == PenaltyTypes.NONE:
            return

        if penalty == PenaltyTypes.L1:
            self.solver.set_proximal_function(
                regularizers.L1Regularizer(
                    penalty_args.get("alpha", DEFAULT_L1_PENALTY)
                )
            )
        elif penalty == PenaltyTypes.L2:
            self.solver.add_gradient_penalty_function(
                regularizers.L2Regularizer(
                    penalty_args.get("alpha", DEFAULT_L2_PENALTY)
                )
            )
        elif penalty == PenaltyTypes.ELASTICNET:
            self._add_penalties(
                PenaltyTypes.L1,
                **{"alpha": penalty_args.get("alpha1", DEFAULT_L1_PENALTY)},
            )
            self._add_penalties(
                PenaltyTypes.L2,
                **{"alpha": penalty_args.get("alpha2", DEFAULT_L2_PENALTY)},
            )
        else:
            raise UnknownPenaltyError(f"Received unknown penalty type: {penalty}")

    @overload
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    @abstractmethod
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        """
        Evaluate the gradient function.

        :param X: Independent data.
        :param y: Dependent data.
        :param coef_: Coefficient vector.
        :param grad_per_sample: Return gradient per sample if True, return aggregated gradient of all data if False.
        :return: Value(s) of gradient evaluated with the provided parameters.
        """

    @abstractmethod
    def score(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Union[Vector[float], Vector[SecureFixedPoint]],
    ) -> SecureFixedPoint:
        """
        Compute the model score.

        :param X: Test data.
        :param y: True value for $X$.
        :param coef_: Coefficient vector.
        :return: Score of the model prediction.
        """

    async def compute_coef_mpc(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        tolerance: float = 1e-2,
        minibatch_size: Optional[int] = None,
        coef_init: Optional[Vector[SecureFixedPoint]] = None,
        nr_maxiters: int = 100,
        eta0: Optional[float] = None,
        print_progress: bool = False,
        secure_permutations: bool = False,
    ) -> Vector[float]:
        """
        Train the model, compute and return the model coefficients.

        :param X: Training data.
        :param y: Target vector.
        :param tolerance: Threshold for convergence.
        :param minibatch_size: The size of the minibatch.
        :param coef_init: Initial coefficient vector to use.
        :param nr_maxiters: Threshold for the number of iterations.
        :param eta0: Initial learning rate.
        :param print_progress: Set to True to print progress every few iterations.
        :param secure_permutations: Set to True to perform matrix permutation securely.
        :raise SecureLearnTypeError: if the training or target data does not consist of secure numbers.
        :return: Coefficient vector.
        """
        start_time = time()

        if not isinstance(X[0][0], SecureNumber):
            raise SecureLearnTypeError(
                f"Elements of the independent data matrix are not of type {repr(SecureNumber)}, but of type {type(X[0][0])}."
            )
        if not isinstance(y[0], SecureNumber):
            raise SecureLearnTypeError(
                f"Elements of the dependent data vector are not of type {repr(SecureNumber)}, but of type {type(y[0])}."
            )

        stype = type(X[0][0])
        self.solver.init_solver(
            total_size=len(X),
            num_features=len(X[0]),
            tolerance=tolerance,
            minibatch_size=minibatch_size,
            coef_init=coef_init,
            sectype=stype,
            eta0=eta0,
        )

        Xperm, yperm = X.copy(), y.copy()

        # Compute coefficients
        coef_mpc = self.solver._get_coefficients(
            Xperm,
            yperm,
            n_maxiter=nr_maxiters,
            print_progress=print_progress,
            secure_permutations=secure_permutations,
        )

        await mpc.barrier()
        coef_ = [float(_) for _ in await mpc.output(coef_mpc)]
        timing = str(time() - start_time)
        logging.info("Timing MPC: " + timing + " seconds")
        logging.info(
            "Number of iterations: "
            + str(self.solver.nr_epochs)
            + "; Relative update difference: "
            + str(self.solver.rel_update_diff)
        )

        return coef_

    async def cross_validate(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        tolerance: float = 1e-2,
        minibatch_size: Optional[int] = None,
        coef_init: Optional[Vector[SecureFixedPoint]] = None,
        nr_maxiters: int = 100,
        eta0: Optional[float] = None,
        print_progress: bool = False,
        secure_permutations: bool = False,
        folds: Union[int, List[Tuple[List[int], List[int]]]] = 5,
        random_state: Optional[int] = None,
        shuffle: bool = False,
    ) -> Vector[float]:
        r"""
        Evaluate metrics over the model prediction using CV.

        :param X: Train data.
        :param y: Target variable for X
        :param tolerance: Threshold for convergence
        :param minibatch_size: The size of the minibatch
        :param coef_init: Initial coefficient vector to use
        :param nr_maxiters: Threshold for the number of iterations
        :param eta0: Initial learning rate
        :param print_progress: Set to True to print progress
        :param secure_permutations: Set to True to perform matrix permutation securely
        :param folds:
            Folding sets.
            If set to $k$ (integer) then a KFold (from sklearn.model_selection) is used. If it is not set then KFold is called with $k=5$.
            It also possible to pass custom folds as a list of tuples of train and test indexes:
            e.g. $[([2, 3], [0, 1, 4]), ([0, 1, 3], [2, 4]), ([0, 1, 2], [3, 4])]$ is a 3-fold of an array of five elements
            $([2, 3] , [0, 1, 4] )$ -> 1st fold, elements with indexes $[2, 3]$ are used in the train set, while elements with indexes  [0, 1, 4]  are used in the test set
            $([0, 1, 3] , [2, 4] )$ -> 2nd fold, elements with indexes $[0, 1, 3]$ are used in the train set, while elements with indexes  [2, 4]  are used in the test set
            $([0, 1, 2] , [3, 4] )$ -> 3rd fold, elements with indexes $[0, 1, 2]$ are used in the train set, while elements with indexes $[3, 4]$ are used in the test set
        :param random_state: parameters that control the randomness of each fold. Pass a value to obtain the same fold each time for reproducibility purposes.
        :param shuffle: Whether to shuffle the data or not before splitting into batches.
        :return: List of scores of the model prediction.
        """
        results = []

        if isinstance(folds, int):
            if not shuffle and random_state is not None:
                random_state = None
            kfold = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            folds = list(kfold.split(X))  # type: ignore[arg-type]

        for train_index, test_index in folds:
            X_train = [X[_] for _ in train_index]
            X_test = [X[_] for _ in test_index]
            y_train = [y[_] for _ in train_index]
            y_test = [y[_] for _ in test_index]

            async with mpc:
                # Train the model
                coef_ = await self.compute_coef_mpc(
                    X_train,
                    y_train,
                    tolerance=tolerance,
                    minibatch_size=minibatch_size,
                    coef_init=coef_init,
                    nr_maxiters=nr_maxiters,
                    eta0=eta0,
                    print_progress=print_progress,
                    secure_permutations=secure_permutations,
                )

                results.append(await mpc.output(self.score(X_test, y_test, coef_)))

            self.solver.permutable_matrix = (
                MatrixAugmenter()
            )  # avoid duplicate key error on following runs
        return results

    @staticmethod
    @abstractmethod
    def predict(
        X: Matrix[SecureFixedPoint],
        coef_: Union[Vector[float], Vector[SecureFixedPoint]],
        **kwargs: Any,
    ) -> List[SecureFixedPoint]:
        """
        Predicts target values for input data.

        :param X: Input data with all features
        :param coef_: Coefficient vector of the model
        :param kwargs: Additional keyword arguments that are needed to predict
        :return: Target values
        """
