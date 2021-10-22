"""
Class for the gradient descent method
"""
from typing import Optional, Tuple

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint

from tno.mpc.mpyc.secure_learning.solvers.solver import Solver
from tno.mpc.mpyc.secure_learning.utils import Matrix, Vector
from tno.mpc.mpyc.secure_learning.utils import util_matrix_vec as mpc_utils


class GD(Solver):
    """
    Class for the gradient descent method
    """

    name = "GD"

    def __init__(self) -> None:
        """
        Constructor method.
        """
        super().__init__()
        self.w_prev: Optional[Vector[SecureFixedPoint]] = None
        self.w_prevprev: Optional[Vector[SecureFixedPoint]] = None

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
        stype = type(X_init[0][0])
        assert self.n is not None

        X = mpc_utils.matrix_transpose(
            [[stype(1, integral=False)] * self.n] + mpc_utils.matrix_transpose(X_init)
        )
        # Copy y: otherwise y is changed in the global environment after
        # permuting!
        y = y_init.copy()

        self.eta0 = 1 / mpc.max(
            [mpc.in_prod(X[_][1:], X[_][1:]) for _ in range(self.n)]
        )

        return X, y

    def inner_loop_calculation(
        self,
        X: Matrix[SecureFixedPoint],
        y: Vector[SecureFixedPoint],
        weights_old: Vector[SecureFixedPoint],
        epoch: int,
    ) -> Vector[SecureFixedPoint]:
        """
        Performs one inner-loop iteration for the solver. Inner-loop refers
        to iteratively looping through the data in batches rather than looping
        over the complete data multiple times.

        :param X: Independent data
        :param y: Dependent data
        :param weights_old: Current iterative solution
        :param epoch: Number of times that the outer loop has completed
        :return: Updated iterative solution
        """
        weights_new = weights_old.copy()
        assert (
            self.minibatch_size is not None
            and self.n is not None
            and self.eta0 is not None
        )

        for index in range(self.nr_inner_iters):
            # Accelerated proximal gradient method, currently non-configurable by user
            accelerated = True
            if accelerated:
                self.w_prevprev = self.w_prev or weights_old
                self.w_prev = weights_old
                v = mpc.vector_add(
                    self.w_prev,
                    [
                        (epoch - 1) / (epoch + 2) * _
                        for _ in mpc.vector_sub(self.w_prev, self.w_prevprev)
                    ],
                )
            else:
                v = weights_old

            # Compute gradient
            unpenalized_gradient_minibatch = (
                self.evaluate_gradient_function_for_minibatch(
                    X[index * self.minibatch_size : (index + 1) * self.minibatch_size],
                    y[index * self.minibatch_size : (index + 1) * self.minibatch_size],
                    v,
                    nr_samples_total=self.n,
                    grad_per_sample=False,
                )
            )

            # Add penalty due to differentiable regularizers
            nr_samples_minibatch = len(
                X[index * self.minibatch_size : (index + 1) * self.minibatch_size]
            )
            gradient_penalty_minibatch = (
                self.compute_aggregated_differentiable_regularizer_penalty(
                    v,
                    nr_samples_minibatch=nr_samples_minibatch,
                    nr_samples_total=self.n,
                )
            )
            penalized_gradient_minibatch = mpc.vector_add(
                unpenalized_gradient_minibatch, gradient_penalty_minibatch
            )

            eta = self.eta0 / (epoch + 1) ** 0.1
            # Convergence guaranteed for eta = O(epoch ** p) where -1 <= p < -.5

            x_k = mpc.vector_sub(v, mpc.scalar_mul(eta, penalized_gradient_minibatch))

            # Evaluate proxy func; add penalty due to non-differentiable
            # regularizers
            if self.has_proximal_function:
                weights_new = self.evaluate_proximal_function(x_k, eta)
            else:
                weights_new = x_k

            weights_old = weights_new.copy()
        return weights_new
