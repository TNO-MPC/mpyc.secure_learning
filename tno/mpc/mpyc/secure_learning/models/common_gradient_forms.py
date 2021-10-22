"""
Provides classes for computing the gradient of objective functions
"""
from typing import Callable, List, Union, cast, overload

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
        weights: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
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
            from the independent data and the model weights
        """
        self.predictive_func = predictive_func

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: Literal[False],
    ) -> Vector[SecureFixedPoint]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: Literal[True],
    ) -> List[Vector[SecureFixedPoint]]:
        ...

    @overload
    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: bool,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        ...

    def __call__(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights: Vector[SecureFixedPoint],
        grad_per_sample: bool = False,
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        """
        Evaluate the gradient from the given parameters.

        Note that this function calculates the gradient as if the input data
        consists out of all data samples. That is, it does not incorporate
        weights for gradients of partial input data.

        :param X: Independent variables
        :param y: Dependent variables
        :param weights: Current weights vector
        :param grad_per_sample: Return a list with gradient per sample instead
            of aggregated (summed) gradient
        :return: Gradient of objective function as specified in class
            docstring, evaluated from the provided parameters
        """
        predicted_labels = self.predictive_func(X, weights)
        prediction_error = mpc.vector_sub(predicted_labels, y)

        if not grad_per_sample:
            return cast(
                Vector[SecureFixedPoint],
                mpc_utils.mat_vec_mult(X, prediction_error, transpose=True),
            )
        return cast(
            Matrix[SecureFixedPoint],
            mpc_utils.mult_scalar_mul(prediction_error, X, transpose=True),
        )
