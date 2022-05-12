"""
Implementation of Linear regression model.
"""

from typing import List, Union, overload

from mpyc.sectypes import SecureFixedPoint
from typing_extensions import Literal

import tno.mpc.mpyc.secure_learning.metrics as sec_metrics
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


class Linear(Model):
    r"""
    Solver for Linear regression. Optimizes a model with objective function:
    $$\frac{1}{{2n}_\textrm{samples}} \times ||y - Xw||^2_2$$

    The gradient is given by:
    $$g(X, y, w) = \frac{1}{{2n}_\textrm{samples}} \times X^T (Xw - y)$$

    See secure_model.py docstrings for more information on solver types
    and penalties.
    """
    name = "Linear regression"

    def __init__(
        self,
        solver_type: SolverTypes = SolverTypes.GD,
        penalty: PenaltyTypes = PenaltyTypes.NONE,
        **penalty_args: float,
    ) -> None:
        """
        Constructor method.

        :param solver_type: Solver type to use (e.g. Gradient Descent aka GD)
        :param penalty: Choose whether using L1, L2 or no penalty
        :param penalty_args: Necessary arguments for chosen penalty.
        """
        super().__init__(solver_type, penalty=penalty, **penalty_args)
        self.weighted_differences_gradient = WeightedDifferencesGradient(
            self._predictive_func
        )

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

    def gradient_function(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Matrix[SecureFixedPoint], Vector[SecureFixedPoint]]:
        """
        Evaluate the gradient from the given parameters.

        :param X: Independent variables
        :param y: Dependent variables
        :param coef_: Current coefficients vector
        :param grad_per_sample: Return a list with gradient per sample
            instead of aggregated (summed) gradient
        :return: Gradient of objective function as specified in class
            docstring, evaluated from the provided parameters
        """
        uncorrected_weighted_differences_gradient = self.weighted_differences_gradient(
            X,
            y,
            coef_,
            grad_per_sample=grad_per_sample,
        )
        return mpc_utils.scale_vector_or_matrix(
            1 / len(X), uncorrected_weighted_differences_gradient
        )

    def score(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Union[Vector[float], Vector[SecureFixedPoint]],
    ) -> SecureFixedPoint:
        """
        Compute the coefficient of determination $R^2$ of the prediction.

        :param X: Test data.
        :param y: True value for $X$.
        :param coef_: Coefficient vector.
        :return: Score of the model prediction.
        """
        predicted_y = self.predict(X, coef_)
        return sec_metrics.r2_score(y, predicted_y)

    @staticmethod
    def _predictive_func(
        X: Matrix[SecureFixedPoint], coef_: Vector[SecureFixedPoint]
    ) -> Vector[SecureFixedPoint]:
        """
        Compute matrix multiplication on matrix $X$ and vector $y$.

        :param X: Independent variables
        :param coef_: Current coefficients vector
        :return: Multiplication of $X$ and coefficients
        """
        return mpc_utils.mat_vec_mult(X, coef_)

    @staticmethod
    def predict(
        X: Matrix[SecureFixedPoint],
        coef_: Union[Vector[float], Vector[SecureFixedPoint]],
        **_kwargs: None,
    ) -> Vector[SecureFixedPoint]:
        """
        Predicts target values for input data to regression model.

        :param X: Input data with all features
        :param coef_: Coefficient vector of regression model
        :param _kwargs: Not used
        :return: Target values of regression model
        """
        stype = type(X[0][0])
        if not isinstance(coef_[0], stype):
            coef_ = [stype(_) for _ in coef_]
        if len(X[0]) == len(coef_):
            return mpc_utils.mat_vec_mult(X, coef_)
        return [xw + coef_[0] for xw in mpc_utils.mat_vec_mult(X, coef_[1:])]
