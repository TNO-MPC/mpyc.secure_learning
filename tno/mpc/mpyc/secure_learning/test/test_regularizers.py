"""
Tests for regularizer module. Purpose is to verify the logic of the
regularizers as they may indicate bugs in the code. Purpose is not to
experimentally validate the precise accuracy of the module.
"""
from typing import Iterable, List

import pytest
from mpyc.runtime import mpc

from tno.mpc.mpyc.secure_learning.regularizers import L1Regularizer, L2Regularizer
from tno.mpc.mpyc.secure_learning.test.mpyc_test_utils import mpyc_input, mpyc_output

pytestmark = [
    pytest.mark.asyncio,
]

# sectypes for general use
SECFXP = mpc.SecFxp()

# Tolerable errors
TOLERABLE_ABS_ERROR = 2 ** -(SECFXP.frac_length // 2)


def l1_regularizer(x: Iterable[float], rate: float, alpha: float) -> List[float]:
    """
    Apply the proximal function for the L1 regularizer. This is done by
    means of the soft-thresholding algorithm.

    :param x: Vector to be regularized
    :param rate: Learning rate of the solver
    :param alpha: Regularization parameter
    :return: Soft-thresholded vector x
    """
    assert rate >= 0

    def soft_thresholding(x: float, eta: float) -> float:
        """
        Apply the soft-thresholding algorithm to a single element.

        :param x: Element of the vector that needs to be soft-thresholded
        :param eta: Soft-thresholding parameter
        :return: Soft-thresholded value of x
        """
        if x > eta:
            return x - eta
        if x < -eta:
            return x + eta
        return 0

    return [soft_thresholding(_, rate * alpha) for _ in x]


def l2_regularizer(x: Iterable[float], alpha: float) -> List[float]:
    r"""
    Compute derivative of the L2 norm
    $\frac{\alpha}{2} ||x||_2^2 \in x : \alpha x$.

    :param x: Vector to be regularized
    :param alpha: Regularization parameter
    :param alpha: float
    :return: Derivative of the L2 norm
    """
    return [alpha * _ for _ in x]


# test data
W_LIST = (
    list(range(-20, 21)),
    [_ / 2 for _ in range(-20, 21)],
    [_ / 10 for _ in range(-20, 21)],
    [0.2, -4, -3, 15, 0.02, -4.3],
)
RATE_LIST = [0, 0.1, 1, 10]
ALPHA_LIST = [0, 0.1, 1, 10]


@pytest.mark.parametrize("weights", W_LIST)
@pytest.mark.parametrize("alpha", ALPHA_LIST)
class TestDifferentiableRegularizer:
    """
    Test differentiable regularizers.
    """

    @staticmethod
    async def test_l2_regularizer(
        weights: List[float],
        alpha: float,
    ) -> None:
        """
        Test for validating the accuracy of the secure L2 regularizer
        implementation.

        :param weights: Weights vector
        :param alpha: Regularization parameter
        """
        secure_weights = mpyc_input(weights, SECFXP)
        secure_l2_regularizer = L2Regularizer(alpha)

        plain_l2_regularized_weights = l2_regularizer(weights, alpha=alpha)
        secure_l2_regularized_weights = secure_l2_regularizer(secure_weights)
        result = await mpyc_output(secure_l2_regularized_weights)
        assert result == pytest.approx(
            plain_l2_regularized_weights, abs=TOLERABLE_ABS_ERROR
        )


@pytest.mark.parametrize("weights", W_LIST)
@pytest.mark.parametrize("rate", RATE_LIST)
@pytest.mark.parametrize("alpha", ALPHA_LIST)
class TestNonDifferentiableRegularizer:
    """
    Test non-differentiable regularizers.
    """

    @staticmethod
    async def test_l1_regularizer(
        weights: List[float],
        rate: float,
        alpha: float,
    ) -> None:
        """
        Test for validating the accuracy of the secure L1 regularizer
        implementation.

        :param weights: Weights vector.
        :param rate: Learning rate of the solver
        :param alpha: Regularization parameter
        """
        secure_weights = mpyc_input(weights, SECFXP)
        secure_rate = mpyc_input(rate, SECFXP)
        secure_l1_regularizer = L1Regularizer(alpha)

        plain_l1_regularized_weights = l1_regularizer(weights, rate=rate, alpha=alpha)
        secure_l1_regularized_weights = secure_l1_regularizer(
            secure_weights, eta=secure_rate
        )
        result = await mpyc_output(secure_l1_regularized_weights)
        assert result == pytest.approx(
            plain_l1_regularized_weights, abs=TOLERABLE_ABS_ERROR
        )
