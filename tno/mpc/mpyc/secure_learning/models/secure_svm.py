"""
Implementation of Support Vector Machine regression model.
"""


from typing import List, Union, cast, overload

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint
from typing_extensions import Literal

from tno.mpc.mpyc.secure_learning.metrics import accuracy_score
from tno.mpc.mpyc.secure_learning.models.secure_model import (
    Model,
    PenaltyTypes,
    SolverTypes,
)
from tno.mpc.mpyc.secure_learning.utils import Matrix, Vector
from tno.mpc.mpyc.secure_learning.utils import util_matrix_vec as mpc_utils


class SVM(Model):
    r"""
    Solver for support vector machine. Optimizes a model with objective
    function:
    $$\frac{1}{n_{\textrm{samples}}} \sum_{i=1}^{n_{\textrm{samples}}} h_w(x_i, y_i).$$

    Here, $h_w(x, y)$ is the hinge loss, defined as:
    $$\max(0, 1 - y \times (w^T x))$$

    Labels $y_i$ are assumed to obtain values $-1$ and $1$.
    """
    name = "SVM"

    def __init__(
        self,
        solver_type: SolverTypes = SolverTypes.GD,
        penalty: PenaltyTypes = PenaltyTypes.L2,
        **penalty_args: float
    ) -> None:
        """
        Constructor method.

        :param solver_type: Solver type to use (e.g. Gradient Descent aka GD)
        :param penalty: Choose whether using L1, L2 or no penalty
        :param penalty_args: Necessary arguments for chosen penalty
        """
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
    ) -> Union[Vector[SecureFixedPoint], List[Vector[SecureFixedPoint]]]:
        r"""
        This function calculates the gradient as if the input data consists
        out of all data samples. That is, it does not incorporate weights for
        gradients of partial input data.

        The gradient itself is given by
        $$1 / n * \sum_{i=1} [0 if y \times (w^T X_i) \geq 1 else y_i * X_i]$$

        :param X: Input matrix
        :param y: Dependent variable
        :param weights: Current weights vector
        :param grad_per_sample: Return a list with gradient per sample instead
            of aggregated (summed) gradient
        :return: Gradient of SVM objective function.
        """
        stype = type(X[0][0])

        sub_cost = mpc.schur_prod(y, mpc_utils.mat_vec_mult(X, weights))
        max_sub_cost = [s < stype(1) for s in sub_cost]
        # For some reason, the vectorized GE is slower? Perhaps only
        # if tno.mpc.mpyc.secure_learning package is installed as editable?
        # max_sub_cost = mpc_utils.vector_ge([stype(1)] * batch_size, sub_cost)
        derivative_per_sample = mpc_utils.mult_scalar_mul(
            mpc.schur_prod(max_sub_cost, y), X, transpose=True
        )
        if not grad_per_sample:
            aggregated_derivative = mpc_utils.matrix_sum(derivative_per_sample)
            return cast(
                Vector[SecureFixedPoint],
                mpc_utils.scale_vector_or_matrix(-1 / len(X), aggregated_derivative),
            )
        return cast(
            Matrix[SecureFixedPoint],
            mpc_utils.scale_vector_or_matrix(-1 / len(X), derivative_per_sample),
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

    @staticmethod
    def predict(
        X: Matrix[SecureFixedPoint],
        weights: Union[Vector[float], Vector[SecureFixedPoint]],
        **_kwargs: None
    ) -> Vector[SecureFixedPoint]:
        """
        Predicts labels for input data to classification model. Label $-1$ is
        assigned of the predicted probability is less then `prob`, otherwise label
        $+1$ is assigned.

        :param X: Input data with all features
        :param weights: Weight vector of classification model
        :param _kwargs: Not used
        :return: Target labels of classification model
        """
        raise NotImplementedError("Prediction for SVM is not yet implemented.")
