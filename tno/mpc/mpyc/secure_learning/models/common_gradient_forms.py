"""
Provides classes for computing the gradient of objective functions
"""
from typing import Callable, List, Optional, Union, cast, overload

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint
from typing_extensions import Literal, Protocol

from tno.mpc.mpyc.secure_learning.utils import Matrix, Vector
from tno.mpc.mpyc.secure_learning.utils import util_matrix_vec as mpc_utils


class GradientFunction(Protocol):
    """
    Class for objective function.
    """

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...


class WeightedDifferencesGradient(GradientFunction):
    """
    Class for computing the gradient of objective functions. The gradient
    is assumed to have the following form:
    $$g(X, y, w) = X^T (f(X, w) - y)$$

    We refer to $f$ as the predictive function.
    """

    def __init__(
        self,
        predictive_func: Callable[
            [Matrix[SecureFixedPoint], Vector[SecureFixedPoint]],
            Vector[SecureFixedPoint],
        ],
    ) -> None:
        """
        Constructor method.

        :param predictive_func: Function that predicts the dependent variable
            from the independent data and the model coefficients
        """
        self.predictive_func = predictive_func

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
        weights_list: Optional[Vector[SecureFixedPoint]] = None,
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
        weights_list: Optional[Vector[SecureFixedPoint]] = None,
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool,
        weights_list: Optional[Vector[SecureFixedPoint]] = None,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        coef_: Vector[SecureFixedPoint],
        grad_per_sample: bool = False,
        weights_list: Optional[Vector[SecureFixedPoint]] = None,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        """
        Evaluate the gradient from the given parameters.

        Note that this function calculates the gradient as if the input data
        consists out of all data samples. That is, it does not incorporate
        coefficients for gradients of partial input data.

        :param X: Independent variables
        :param y: Dependent variables
        :param coef_: Current coefficients vector
        :param grad_per_sample: Return a list with gradient per sample instead
            of aggregated (summed) gradient
        :param weights_list: List of class weights to scale the prediction error
            by, defaults to None
        :return: Gradient of objective function as specified in class
            docstring, evaluated from the provided parameters
        """
        predicted_labels = self.predictive_func(X, coef_)
        prediction_error = mpc.vector_sub(predicted_labels, y)

        if weights_list is not None:
            weighted_prediction_error = mpc.schur_prod(weights_list, prediction_error)
        else:
            weighted_prediction_error = prediction_error

        if not grad_per_sample:
            return cast(
                Vector[SecureFixedPoint],
                mpc_utils.mat_vec_mult(X, weighted_prediction_error, transpose=True),
            )
        return cast(
            Matrix[SecureFixedPoint],
            mpc_utils.mult_scalar_mul(weighted_prediction_error, X, transpose=True),
        )
