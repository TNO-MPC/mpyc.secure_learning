"""
Contains penalty functions.
"""
from abc import ABC, abstractmethod
from typing import List, Union

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint


class BaseRegularizer:
    """
    Base class for regularizations.
    """


class DifferentiableRegularizer(ABC, BaseRegularizer):
    """
    Differentiable regularizations can be included via their gradient.
    """

    @abstractmethod
    def __call__(self, coef_: List[SecureFixedPoint]) -> List[SecureFixedPoint]:
        """
        Evaluate the regularization gradient function.

        :param coef_: Coefficient vector
        :return: Value of regularization gradient function evaluated with the
            provided parameters.
        """


class L2Regularizer(DifferentiableRegularizer):
    r"""
    Implementation for L2 regularization:
    $$f(w) = \frac{\alpha}{2} \times ||w||^2_2`.$$
    """

    def __init__(self, alpha: float):
        """
        Constructor method.

        :param alpha: Regularization parameter
        """
        self.alpha = alpha

    def __call__(self, coef_: List[SecureFixedPoint]) -> List[SecureFixedPoint]:
        """
        Apply the initialized L2 regularization.

        :param coef_: Coefficient vector to be regularized.
        :return: Value of regularization gradient evaluated with the provided
            parameters.
        """
        return [self.alpha * _ for _ in coef_]


class NonDifferentiableRegularizer(ABC, BaseRegularizer):
    """
    Non-differential regularization can be included via a proximal
    method.
    """

    @abstractmethod
    def __call__(
        self, coef_: List[SecureFixedPoint], eta: Union[float, SecureFixedPoint]
    ) -> List[SecureFixedPoint]:
        """
        Apply the proximal function for this regularizer.

        :param coef_: Coefficient vector.
        :param eta: Learning rate.
        :return: Value of proximal function evaluated with the provided
            parameters.
        """


class L1Regularizer(NonDifferentiableRegularizer):
    r"""
    Implementation for L1 regularization: $f(w) = ||w||_1$.
    """

    def __init__(self, alpha: float):
        """
        Constructor method.

        :param alpha: Regularization parameter
        """
        self.alpha = alpha

    def __call__(
        self, coef_: List[SecureFixedPoint], eta: Union[float, SecureFixedPoint]
    ) -> List[SecureFixedPoint]:
        r"""
        Apply the proximal function for the L1 regularizer.

        This proximal function is more commonly known as the soft-thresholding
        algorithm. The soft-thresholding algorithm pulls every element of
        $w$ (coefficient vector) closer to zero. It does so in a component-wise
        fashion. More specifically:
        $$
        \textrm{new\_}w_i = \left\{ \begin{array}{cl}
        w_i - \nu & : \ w_i > \nu \\
        0 & : \ -\nu < w_i < \nu \\
        w_i + \nu  & : \ -\nu < w_i
        \end{array} \right.
        $$

        Here, $\nu$ is a value that depends on eta and the regularization
        constant $\alpha$.

        :param coef_: Coefficient vector.
        :param eta: Learning rate.
        :return: Value of proximal function evaluated with the provided
            parameters.
        """
        stype = type(coef_[0])

        nu = eta * self.alpha

        # Find sign and absolute value of xi
        # Since mpc.sgn(stype(0)) = 0,
        # instead we compute 2*mpc.ge(stype(0), 0) - 1
        signs = [2 * mpc.ge(_, stype(0)) - 1 for _ in coef_]
        abs_coef_ = mpc.schur_prod(signs, coef_)

        #       = xi - nu if       xi > nu
        # new_x = 0       if -nu < xi < nu
        #       = xi + nu if -nu < xi
        xi_gtr_nu = [mpc.ge(_, nu) for _ in abs_coef_]
        new_coef_ = mpc.schur_prod(
            xi_gtr_nu,
            mpc.vector_sub(coef_, mpc.scalar_mul(nu, signs)),
        )
        return new_coef_
