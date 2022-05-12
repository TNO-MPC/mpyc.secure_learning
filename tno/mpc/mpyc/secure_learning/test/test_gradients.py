"""
Tests for metrics module. Purpose is to identify large deviations from the
expected (sklearn) metrics as they may indicate bugs in the code. Purpose is
not to experimentally validate the precise accuracy of the metrics module.
"""
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
from mpyc.runtime import mpc

from tno.mpc.mpyc.secure_learning import SVM, Linear, Logistic
from tno.mpc.mpyc.secure_learning.models import ExponentiationTypes, PenaltyTypes
from tno.mpc.mpyc.secure_learning.test.mpyc_test_utils import mpyc_input, mpyc_output
from tno.mpc.mpyc.secure_learning.test.plaintext_utils.plaintext_gradients import (
    plain_approximate_logistic_gradient,
    plain_exact_logistic_gradient,
    plain_linear_gradient,
    plain_svm_gradient,
)
from tno.mpc.mpyc.secure_learning.utils import Matrix, Vector

pytestmark = [
    pytest.mark.asyncio,
]

# sectypes for general use
SECFXP = mpc.SecFxp()

# Tolerable errors. Either the relative OR the absolute errors must be met
# in tests.
TOLERABLE_ABS_ERROR = 2 ** -(SECFXP.frac_length // 2)
TOLERABLE_REL_ERROR = 2 ** -(SECFXP.frac_length // 2)


# test data
# data should be somewhat normalized, but also include sets with smaller
# values for classification gradient testing
DATA_LIST = (
    list(list(i + _ for _ in range(-10, 11)) for i in range(-20, 21)),
    list(
        list((-1) ** i * (2 * i + _) % 19 for _ in range(-20, 21))
        for i in range(-20, 21)
    ),
    list(list(0.01 * (i + _) for _ in range(-30, 31)) for i in range(-20, 21)),
)
COEF_LIST = (
    list(_ / 10 for _ in range(-10, 11)),
    list(-2 * _ for _ in range(-20, 21)),
    list(_ / 60 for _ in range(-30, 31)),
)
REGRESSION_LABELS_LIST = (
    list(_ / 10 for _ in range(-20, 21)),
    list(-_ / 6 for _ in range(-20, 21)),
    list(2 * _ for _ in range(-20, 21)),
)
CLASSIFICATION_LABELS_LIST = (
    list(i % 2 for i in range(-20, 21)),
    list(-1 if i < 0 else 1 for i in range(-20, 21)),
    list(-1 if abs(i) > 15 else 1 for i in range(-20, 21)),
)


PYTEST_MARK_PARAMETRIZE_REGRESSION = pytest.mark.parametrize(
    "data, coef_, labels",
    list(
        zipped_input
        for zipped_input in zip(DATA_LIST, COEF_LIST, REGRESSION_LABELS_LIST)
    ),
)

PYTEST_MARK_PARAMETRIZE_CLASSIFICATION = pytest.mark.parametrize(
    "data, coef_, labels",
    list(
        zipped_input
        for zipped_input in zip(DATA_LIST, COEF_LIST, CLASSIFICATION_LABELS_LIST)
    ),
)


@dataclass
class GradientFunctions:
    """
    Class for objective function.
    """

    plain: Callable[..., Any]
    secure: Callable[..., Any]


class BaseTestGradient:
    """
    Test for computing the gradient for regression models.
    """

    async def test_correctness_per_sample(
        self,
        gradient: GradientFunctions,
        data: Matrix[float],
        coef_: Vector[float],
        labels: Vector[float],
    ) -> None:
        """
        Test for validating the accuracy of the gradient if it is computed
        per sample.

        :param gradient: The gradient to be tested
        :param data: Independent input data
        :param coef_: Coefficients vector
        :param labels: Real labels corresponding to independent input data
        """
        plain_gradient = gradient.plain(
            X=data, y=labels, coef_=coef_, grad_per_sample=True
        )

        secure_data = mpyc_input(data, SECFXP)
        secure_coef_ = mpyc_input(coef_, SECFXP)
        secure_labels = mpyc_input(labels, SECFXP)

        secure_gradient = gradient.secure(
            X=secure_data, y=secure_labels, coef_=secure_coef_, grad_per_sample=True
        )

        result = await mpyc_output(secure_gradient)
        assert all(
            secure_res
            == pytest.approx(
                plain_res, abs=TOLERABLE_ABS_ERROR, rel=TOLERABLE_REL_ERROR
            )
            for (secure_res, plain_res) in zip(result, plain_gradient)
        )

    async def test_correctness_per_batch(
        self,
        gradient: GradientFunctions,
        data: Matrix[float],
        coef_: Vector[float],
        labels: Vector[float],
    ) -> None:
        """
        Test for validating the accuracy of the gradient if it is computed
        for the full input data (batch).

        :param gradient: The gradient to be tested
        :param data: Independent input data
        :param coef_: Coefficients vector
        :param labels: Real labels corresponding to independent input data
        """
        plain_gradient = gradient.plain(
            X=data, y=labels, coef_=coef_, grad_per_sample=False
        )

        secure_data = mpyc_input(data, SECFXP)
        secure_coef_ = mpyc_input(coef_, SECFXP)
        secure_labels = mpyc_input(labels, SECFXP)

        secure_gradient = gradient.secure(
            X=secure_data,
            y=secure_labels,
            coef_=secure_coef_,
            grad_per_sample=False,
        )

        result = await mpyc_output(secure_gradient)
        assert result == pytest.approx(
            plain_gradient, abs=TOLERABLE_ABS_ERROR, rel=TOLERABLE_REL_ERROR
        )

    async def test_consistency_per_sample_or_batch(
        self,
        gradient: GradientFunctions,
        data: Matrix[float],
        coef_: Vector[float],
        labels: Vector[float],
    ) -> None:
        """
        Test for validating the that computing the gradient per batch is
        equivalent to aggregating the gradients that are computed per sample.

        :param gradient: The gradient to be tested
        :param data: Independent input data
        :param coef_: Coefficients vector
        :param labels: Real labels corresponding to independent input data
        """
        secure_data = mpyc_input(data, SECFXP)
        secure_coef_ = mpyc_input(coef_, SECFXP)
        secure_labels = mpyc_input(labels, SECFXP)

        secure_gradient_per_sample = gradient.secure(
            X=secure_data, y=secure_labels, coef_=secure_coef_, grad_per_sample=True
        )
        secure_gradient_per_batch = gradient.secure(
            X=secure_data,
            y=secure_labels,
            coef_=secure_coef_,
            grad_per_sample=False,
        )

        result_per_sample = await mpyc_output(secure_gradient_per_sample)
        result_per_sample_aggr = np.asarray(result_per_sample).sum(axis=0)
        result_per_batch = await mpyc_output(secure_gradient_per_batch)
        assert result_per_sample_aggr == pytest.approx(
            result_per_batch, abs=TOLERABLE_ABS_ERROR, rel=TOLERABLE_REL_ERROR
        )


@pytest.mark.parametrize(
    "gradient",
    [
        GradientFunctions(
            plain=plain_linear_gradient,
            secure=Linear(penalty=PenaltyTypes.NONE).gradient_function,
        )
    ],
)
@PYTEST_MARK_PARAMETRIZE_REGRESSION
class TestLinearGradient(BaseTestGradient):
    """
    Test for computing the gradient when using the linear model
    """


@pytest.mark.parametrize(
    "gradient",
    [
        GradientFunctions(
            plain=plain_exact_logistic_gradient,
            secure=Logistic(
                penalty=PenaltyTypes.NONE, exponentiation=ExponentiationTypes.EXACT
            ).gradient_function,
        )
    ],
)
@PYTEST_MARK_PARAMETRIZE_CLASSIFICATION
class TestLogisticExactGradient(BaseTestGradient):
    """
    Test for computing the gradient when using the logistic model
    with exact exponentiation
    """


@pytest.mark.parametrize(
    "gradient",
    [
        GradientFunctions(
            plain=plain_approximate_logistic_gradient,
            secure=Logistic(
                penalty=PenaltyTypes.NONE, exponentiation=ExponentiationTypes.APPROX
            ).gradient_function,
        )
    ],
)
@PYTEST_MARK_PARAMETRIZE_CLASSIFICATION
class TestLogisticApproximateGradient(BaseTestGradient):
    """
    Test for computing the gradient when using the logistic model
    with approximated exponentiation
    """


@pytest.mark.parametrize(
    "gradient",
    [
        GradientFunctions(
            plain=plain_svm_gradient,
            secure=SVM(penalty=PenaltyTypes.NONE).gradient_function,
        )
    ],
)
@PYTEST_MARK_PARAMETRIZE_CLASSIFICATION
class TestSVMGradient(BaseTestGradient):
    """
    Test for computing the gradient when using the SVM model
    """
