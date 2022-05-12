"""
Module for Cross Validation (CV) following the GridSearchCV paradigm of sklearn.
"""
import statistics
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Type, TypeVar, Union

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint
from sklearn.model_selection import ParameterGrid

from tno.mpc.mpyc.secure_learning.models import (
    ExponentiationTypes,
    Lasso,
    Logistic,
    PenaltyTypes,
    Ridge,
    SolverTypes,
)
from tno.mpc.mpyc.secure_learning.models.secure_logistic import ClassWeightsTypes
from tno.mpc.mpyc.secure_learning.models.secure_model import Model
from tno.mpc.mpyc.secure_learning.utils import Matrix, Vector

ModelTypeTV = TypeVar("ModelTypeTV", bound=Type[Model])


@dataclass
class ParameterCollection:
    """
    A collection of parameters, serves as a helper class for generation of a parameter grid.
    """

    solver_type: List[SolverTypes]
    penalty: List[PenaltyTypes]
    alpha: List[float]
    tolerance: List[float]
    minibatch_size: List[Optional[int]]
    coef_init: List[Optional[Vector[SecureFixedPoint]]]
    nr_maxiters: List[int]
    class_weights: List[ClassWeightsTypes] = field(
        default_factory=lambda: [ClassWeightsTypes.EQUAL]
    )
    print_progress: List[bool] = field(default_factory=lambda: [False])
    secure_permutations: List[bool] = field(default_factory=lambda: [False])
    exponentiation: List[ExponentiationTypes] = field(
        default_factory=lambda: [ExponentiationTypes.NONE]
    )
    eta0: List[Optional[float]] = field(default_factory=lambda: [None])


@dataclass
class Parameters:
    """
    Helper class, used internally, to provide a set a parameters.
    """

    solver_type: SolverTypes
    penalty: PenaltyTypes
    alpha: float
    tolerance: float
    minibatch_size: Optional[int]
    coef_init: Optional[Vector[SecureFixedPoint]]
    nr_maxiters: int
    class_weights: ClassWeightsTypes
    print_progress: bool
    secure_permutations: bool
    exponentiation: ExponentiationTypes
    eta0: Optional[float]


class GridCV:
    """
    Exhaustive search over specified parameter values for a model.
    """

    def __init__(
        self,
        model_type: ModelTypeTV,
        parameter_collection: ParameterCollection,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        random_state: Optional[int] = None,
        shuffle: bool = False,
    ) -> None:
        """
        Constructor method.

        :param model_type: Model type (class)
        :param parameter_collection: Parameters to use for cross validation
        :param X: Secret-shared training data
        :param y: Secret-shared target vector
        :raise AttributeError: if the parameters contain an unknown key
        """
        super().__init__()
        self.model_type = model_type
        self.random_state = random_state
        self.shuffle = shuffle
        self._X = X
        self._y = y
        self._results: List[Vector[float]] = []
        self._param_grid: List[Parameters] = []
        # create all parameters sets
        for parameters in ParameterGrid(asdict(parameter_collection)):
            self._param_grid.append(Parameters(**parameters))

    @property
    def results(self) -> List[Vector[float]]:
        """
        The results of grid cross-validation

        :return: Results of grid cross-validation
        :raise ValueError: Raised when results are not available
        """
        if not self._results:
            raise ValueError("No results are available yet.")
        return self._results

    async def cross_validation(
        self, folds: Union[int, List[Tuple[List[int], List[int]]]]
    ) -> List[Vector[float]]:
        r"""
        Compute cross validation over all given combinations of parameters and return the score for each set.

        :param folds:
            Folding sets.
            If set to $k$ (integer) then a KFold (from sklearn.model_selection) is used. If it is not set then KFold is called with $k=5$.
            It also possible to pass custom folds as a list of tuples of train and test indexes:
            e.g. $[([2, 3], [0, 1, 4]), ([0, 1, 3], [2, 4]), ([0, 1, 2], [3, 4])]$ is a three-fold of an array of five elements
            $([2, 3], [0, 1, 4])$ -> 1st fold, elements with indexes $[2, 3]$ are used in the train set, while elements with indexes $[0, 1, 4]$ are used in the test set
            $([0, 1, 3], [2, 4])$ -> 2nd fold, elements with indexes $[0, 1, 3]$ are used in the train set, while elements with indexes $[2, 4]$ are used in the test set
            $([0, 1, 2], [3, 4])$ -> 3rd fold, elements with indexes $[0, 1, 2]$ are used in the train set, while elements with indexes $[3, 4]$ are used in the test set
        :return: A list containing one list of score results (over different folds) for each parameters combination
        """
        results = []
        for parameters in self._param_grid:
            if issubclass(self.model_type, Logistic):
                model: Model = self.model_type(
                    solver_type=parameters.solver_type,
                    penalty=parameters.penalty,
                    exponentiation=parameters.exponentiation,
                    class_weights_type=parameters.class_weights,
                    alpha=parameters.alpha,
                )
            elif issubclass(self.model_type, Ridge) or issubclass(
                self.model_type, Lasso
            ):
                model = self.model_type(
                    solver_type=parameters.solver_type,
                    alpha=parameters.alpha,
                )
            else:
                model = self.model_type(
                    solver_type=parameters.solver_type,
                    penalty=parameters.penalty,
                    alpha=parameters.alpha,
                )
            async with mpc:
                result = await model.cross_validate(
                    self._X,
                    self._y,
                    tolerance=parameters.tolerance,
                    minibatch_size=parameters.minibatch_size,
                    coef_init=parameters.coef_init,
                    nr_maxiters=parameters.nr_maxiters,
                    eta0=parameters.eta0,
                    print_progress=parameters.print_progress,
                    secure_permutations=parameters.secure_permutations,
                    folds=folds,
                    random_state=self.random_state,
                    shuffle=self.shuffle,
                )
                results.append(result)
        self._results = results
        return results

    def best_param_set(self) -> Tuple[float, Parameters]:
        """
        Return parameter set with the best score result.

        :return: parameter set with the best score result
        """
        avg = [statistics.mean(result) for result in self._results]
        best = avg.index(max(avg))
        return avg[best], self._param_grid[best]

    def sorted_param_set(self) -> List[Tuple[float, Parameters]]:
        """
        Return a sorted list of the parameter sets with their score results

        :return: sorted list of the parameter sets with their score results
        """
        avg = [
            (statistics.mean(result), self._param_grid[index])
            for index, result in enumerate(self._results)
        ]
        avg = sorted(avg, key=lambda x: (x[0]))
        return avg
