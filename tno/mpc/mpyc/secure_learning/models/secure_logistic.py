"""
Implementation of Logistic regression model.
"""
import math
from enum import Enum, auto
from typing import List, Union, overload

from mpyc.sectypes import SecureFixedPoint
from typing_extensions import Literal

from tno.mpc.mpyc.exponentiation import secure_pow

from tno.mpc.mpyc.secure_learning.metrics import accuracy_score
from tno.mpc.mpyc.secure_learning.models.common_gradient_forms import (
    WeightedDifferencesGradient,
)
from tno.mpc.mpyc.secure_learning.models.secure_model import (
    Model,
    PenaltyTypes,
    SolverTypes,
)
from tno.mpc.mpyc.secure_learning.utils import Matrix, Vector
from tno.mpc.mpyc.secure_learning.utils import util_matrix_vec as mpc_utils


class ExponentiationTypes(Enum):
    """
    Class to store whether exponentations are approximated
    or calculated exactly.
    """

    NONE = auto()
    APPROX = auto()
    EXACT = auto()


class Logistic(Model):
    r"""
    Solver for logistic regression. Optimizes a model with objective function
    $$\left(\frac{1}{2{n}_{\textrm{samples}}}\right) \sum_{i=1}^{{n}_{\textrm{samples}}}\left(\textrm{-}(1+y_i) \log(h_w(x_i)) - (1-y_i) \log(1-h_w(x_i))\right)$$

    Here,
    $$h_w(x) = \frac{1}{(1 + e^{-w^T x}}$$

    Labels $y_i$ are assumed to have value $-1$ or $1$.

    The gradient is given by:
    $$g(X, y, w) = \left(\frac{1}{2} \times {n}_{\textrm{samples}}\right) \sum_{i=1}^{{n}_\textrm{samples}} x_i^T \left( (2h_w(x_i) - 1) - y \right)$$

    See secure_model.py docstrings for more information on solver types and
    penalties.
    """
    name = "Logistic regression"

    def __init__(
        self,
        solver_type: SolverTypes = SolverTypes.GD,
        exponentiation: ExponentiationTypes = ExponentiationTypes.EXACT,
        penalty: PenaltyTypes = PenaltyTypes.NONE,
        **penalty_args: float,
    ) -> None:
        """
        Constructor method.

        :param solver_type: Solver type to use (e.g. Gradient Descent aka GD)
        :param penalty: Choose whether using L1, L2 or no penalty
        :param exponentiation: Choose whether exponentiations are approximated
            or exactly calculated
        :param penalty_args: Necessary arguments for chosen penalty
        :raise ValueError: raised when exponentiation is of wrong type.
        """
        self.name += f" {exponentiation.name}"
        if exponentiation == ExponentiationTypes.EXACT:
            self.weighted_differences_gradient = WeightedDifferencesGradient(
                predictive_func=self._predictive_func_exact
            )
        elif exponentiation == ExponentiationTypes.APPROX:
            self.weighted_differences_gradient = WeightedDifferencesGradient(
                predictive_func=self._predictive_func_approx
            )
        else:
            raise ValueError(
                f"Expected exponentiation in ({ExponentiationTypes.EXACT, ExponentiationTypes.APPROX}), not {exponentiation}."
            )
        super().__init__(solver_type, penalty=penalty, **penalty_args)

    @overload
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Matrix[SecureFixedPoint], Vector[SecureFixedPoint]]:
        """
        Evaluate the gradient from the given parameters.

        :param X: Independent variables
        :param y: Dependent variables
        :param weights: Current weights vector
        :param grad_per_sample: Return a list with gradient per sample instead
            of aggregated (summed) gradient
        :return: Gradient of objective function as specified in class
            docstring, evaluated from the provided parameters
        """
        uncorrected_weighted_differences_gradient = self.weighted_differences_gradient(
            X,
            y,
            weights,
            grad_per_sample=grad_per_sample,
        )
        return mpc_utils.scale_vector_or_matrix(
            1 / (2 * len(X)), uncorrected_weighted_differences_gradient
        )

    def score(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Union[Vector[float], Vector[SecureFixedPoint]],
    ) -> SecureFixedPoint:
        """
        Compute the mean accuracy of the prediction.

        :param X: Test data.
        :param y: True label for $X$.
        :param weights: Weight vector.
        :return: Score of the model prediction.
        """
        predicted_labels = self.predict(X, weights)
        return accuracy_score(y, predicted_labels)

    # self.gradfunc(X, w) is presented / approximated here along the lines of
    # Mohassel and Zhang (2017), "SecureML: A System for Scalable Privacy-
    # Preserving Machine Learning", DOI 10.1109/SP.2017.12
    @staticmethod
    def _predictive_func_approx(
        X: Matrix[SecureFixedPoint], weights: Vector[SecureFixedPoint]
    ) -> Vector[SecureFixedPoint]:
        """
        Compute the predictive function (approximately).

        :param X: Independent variables
        :param weights: Current weights vector
        :return: Output of the predictive function
        """
        stype = type(X[0][0])
        X_times_w = mpc_utils.mat_vec_mult(X, weights)

        d_0 = [x < stype(-0.5) for x in X_times_w]
        d_1 = [x > stype(0.5) for x in X_times_w]

        new_values = [
            d_1[i]
            + (stype(1) - d_0[i]) * (stype(1) - d_1[i]) * (X_times_w[i] + stype(0.5))
            for i in range(len(d_0))
        ]
        # Map predicted {0, 1}-labels to {-1, +1}
        return [2 * val - 1 for val in new_values]

    @staticmethod
    def _predictive_func_exact(
        X: Matrix[SecureFixedPoint], weights: Vector[SecureFixedPoint]
    ) -> Vector[SecureFixedPoint]:
        """
        Compute the predictive function (exactly).

        :param X: Independent variables
        :param weights: Current weights vector
        :return: Output of the predictive function
        """
        X_times_w = mpc_utils.mat_vec_mult(X, weights)
        X_times_w = [x * (-1) for x in X_times_w]

        result = secure_pow(math.e, X_times_w, trunc_to_domain=True, bits_buffer=1)
        new_results = [1 / (1 + r) for r in result]
        # Map predicted {0, 1}-labels to {-1, +1}
        return [2 * val - 1 for val in new_results]

    @staticmethod
    def predict(
        X: Matrix[SecureFixedPoint],
        weights: Union[Vector[float], Vector[SecureFixedPoint]],
        prob: float = 0.5,
        **_kwargs: None,
    ) -> Vector[SecureFixedPoint]:
        """
        Predicts labels for input data to classification model. Label $-1$ is
        assigned of the predicted probability is less then `prob`, otherwise label
        $+1$ is assigned.

        :param X: Input data with all features
        :param weights: Weight vector of classification model
        :param prob: Threshold for labelling. Defaults to $0.5$.
        :param _kwargs: Not used
        :return: Target labels of classification model
        """
        stype = type(X[0][0])
        exp_threshold = math.log(prob / (1 - prob))
        if not isinstance(weights[0], stype):
            weights = [stype(_) for _ in weights]
        exponents = [x + weights[0] for x in mpc_utils.mat_vec_mult(X, weights[1:])]
        return [2 * (x >= exp_threshold) - 1 for x in exponents]
