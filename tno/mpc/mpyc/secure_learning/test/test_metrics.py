"""
Tests for metrics module. Purpose is to identify large deviations from the
expected (sklearn) metrics as they may indicate bugs in the code. Purpose is
not to experimentally validate the precise accuracy of the metrics module.
"""
import numpy as np
import pytest
from mpyc.runtime import mpc

from tno.mpc.mpyc.secure_learning import metrics
from tno.mpc.mpyc.secure_learning.test.mpyc_test_utils import mpyc_input, mpyc_output
from tno.mpc.mpyc.secure_learning.test.plaintext_utils import plaintext_metrics
from tno.mpc.mpyc.secure_learning.utils import NumpyNumberArray, Vector

pytestmark = [
    pytest.mark.asyncio,
]

# sectypes for general use
SECFXP = mpc.SecFxp()

# Tolerable errors
# Metrics that usually yield values in [0, 1]
TOLERABLE_ABS_ERROR = 2 ** -(SECFXP.frac_length // 2)
# Metrics that return some squared difference
## Errors in metrics with squared differences accumulate. Difference vector
## has doubled error, this new error gets squared.
TOLERABLE_ABS_ERROR_SQUARED = 4 * TOLERABLE_ABS_ERROR


# test data
classification_labels = (-1, 1)
_classification_labels_real: NumpyNumberArray = np.concatenate(
    (np.repeat([1], 50), np.repeat([-1], 50))
)
_classification_labels_pred: NumpyNumberArray = np.concatenate(
    (np.repeat([1], 20), np.repeat([-1], 50), np.repeat([1], 30))
)
classification_test_data = list(
    (real, pred)
    for (real, pred) in zip(
        (_classification_labels_real,), (_classification_labels_pred,)
    )
)

_regression_labels_real = np.arange(-10, 10 + 1e-6, step=1 / 10)
nr_rlr = len(_regression_labels_real)
regression_labels_deviations = np.concatenate(
    (
        np.tile([-1, 3, 0, -0.1], nr_rlr // 8),
        np.tile([-0.5, 0, 0, 1.4], nr_rlr + 7 // 8),
    )
)[:nr_rlr]
_regression_labels_pred = _regression_labels_real + regression_labels_deviations
regression_test_data = list(
    (real, pred)
    for (real, pred) in zip((_regression_labels_real,), (_regression_labels_pred,))
)


@pytest.mark.parametrize(
    "classification_labels_real, classification_labels_pred",
    classification_test_data,
)
class TestClassificationMetricsWithoutPosLabel:
    """
    Test classification metrics that are label-agnostic.
    """

    @staticmethod
    async def test_accuracy_score(
        classification_labels_real: Vector[float],
        classification_labels_pred: Vector[float],
    ) -> None:
        """
        Test for validating the accuracy of the secure accuracy_score
        implementation versus sklearn.

        :param classification_labels_real: Real labels
        :param classification_labels_pred: Predicted labels
        """
        secure_classification_labels_real = mpyc_input(
            classification_labels_real, SECFXP
        )
        secure_classification_labels_pred = mpyc_input(
            classification_labels_pred, SECFXP
        )

        plain_accuracy_score = plaintext_metrics.accuracy_score(
            classification_labels_real, classification_labels_pred
        )
        secure_accuracy_score = metrics.accuracy_score(
            secure_classification_labels_real, secure_classification_labels_pred
        )
        result = await mpyc_output(secure_accuracy_score)
        assert result == pytest.approx(plain_accuracy_score, abs=TOLERABLE_ABS_ERROR)


@pytest.mark.parametrize(
    "classification_labels_real, classification_labels_pred",
    classification_test_data,
)
@pytest.mark.parametrize("pos_label", classification_labels)
class TestClassificationMetricsWithPosLabel:
    """
    Test classification metrics that depend on a target label.
    """

    @staticmethod
    async def test_precision_score(
        classification_labels_real: Vector[float],
        classification_labels_pred: Vector[float],
        pos_label: int,
    ) -> None:
        """
        Test for validating the accuracy of the secure precision_score
        implementation versus sklearn.

        :param classification_labels_real: Real labels
        :param classification_labels_pred: Predicted labels
        :param pos_label: Label that is benchmarked
        """
        secure_classification_labels_real = mpyc_input(
            classification_labels_real, SECFXP
        )
        secure_classification_labels_pred = mpyc_input(
            classification_labels_pred, SECFXP
        )

        plain_precision_score = plaintext_metrics.precision_score(
            classification_labels_real, classification_labels_pred, pos_label
        )
        secure_precision_score = metrics.precision_score(
            secure_classification_labels_real,
            secure_classification_labels_pred,
            pos_label,
        )
        result = await mpyc_output(secure_precision_score)
        assert result == pytest.approx(plain_precision_score, abs=TOLERABLE_ABS_ERROR)

    @staticmethod
    async def test_recall_score(
        classification_labels_real: Vector[float],
        classification_labels_pred: Vector[float],
        pos_label: int,
    ) -> None:
        """
        Test for validating the accuracy of the secure recall_score
        implementation versus sklearn.

        :param classification_labels_real: Real labels
        :param classification_labels_pred: Predicted labels
        :param pos_label: Label that is benchmarked
        """
        secure_classification_labels_real = mpyc_input(
            classification_labels_real, SECFXP
        )
        secure_classification_labels_pred = mpyc_input(
            classification_labels_pred, SECFXP
        )

        plain_recall_score = plaintext_metrics.recall_score(
            classification_labels_real, classification_labels_pred, pos_label
        )
        secure_recall_score = metrics.recall_score(
            secure_classification_labels_real,
            secure_classification_labels_pred,
            pos_label,
        )
        result = await mpyc_output(secure_recall_score)
        assert result == pytest.approx(plain_recall_score, abs=TOLERABLE_ABS_ERROR)

    @staticmethod
    async def test_f1_score(
        classification_labels_real: Vector[float],
        classification_labels_pred: Vector[float],
        pos_label: int,
    ) -> None:
        """
        Test for validating the accuracy of the secure f1_score
        implementation versus sklearn.

        :param classification_labels_real: Real labels
        :param classification_labels_pred: Predicted labels
        :param pos_label: Label that is benchmarked
        """
        secure_classification_labels_real = mpyc_input(
            classification_labels_real, SECFXP
        )
        secure_classification_labels_pred = mpyc_input(
            classification_labels_pred, SECFXP
        )

        plain_f1_score = plaintext_metrics.f1_score(
            classification_labels_real, classification_labels_pred, pos_label
        )
        secure_f1_score = metrics.f1_score(
            secure_classification_labels_real,
            secure_classification_labels_pred,
            pos_label,
        )
        result = await mpyc_output(secure_f1_score)
        assert result == pytest.approx(plain_f1_score, abs=TOLERABLE_ABS_ERROR)


@pytest.mark.parametrize("regression_labels_real", (_regression_labels_real,))
class TestRegressionMetricsRealLabelOnly:
    """
    Test regression metrics that do not depend on only one set of labels.
    """

    @staticmethod
    async def test_mean_squared_model(regression_labels_real: Vector[float]) -> None:
        """
        Test for validating the accuracy of the secure mean_squared_model
        implementation versus sklearn.

        :param regression_labels_real: Real labels
        """
        secure_regression_labels_real = mpyc_input(regression_labels_real, SECFXP)

        plain_msm = plaintext_metrics.mean_squared_model(regression_labels_real)
        secure_msm = metrics.mean_squared_model(secure_regression_labels_real)
        result = await mpyc_output(secure_msm)
        assert result == pytest.approx(plain_msm, abs=TOLERABLE_ABS_ERROR_SQUARED)


@pytest.mark.parametrize(
    "regression_labels_real, regression_labels_pred", regression_test_data
)
class TestRegressionMetricsRealAndPredictedLabel:
    """
    Test regression metrics that do not depend on a set of real and a set
    of predicted labels.
    """

    @staticmethod
    async def test_mean_squared_error(
        regression_labels_real: Vector[float],
        regression_labels_pred: Vector[float],
    ) -> None:
        """
        Test for validating the accuracy of the secure mean_squared_error
        implementation versus sklearn.

        :param regression_labels_real: Real labels
        :param regression_labels_pred: Predicted labels
        """
        secure_regression_labels_real = mpyc_input(regression_labels_real, SECFXP)
        secure_regression_labels_pred = mpyc_input(regression_labels_pred, SECFXP)

        plain_mse = plaintext_metrics.mean_squared_error(
            regression_labels_real, regression_labels_pred
        )
        secure_mse = metrics.mean_squared_error(
            secure_regression_labels_real, secure_regression_labels_pred
        )
        result = await mpyc_output(secure_mse)
        assert result == pytest.approx(plain_mse, abs=TOLERABLE_ABS_ERROR_SQUARED)

    @staticmethod
    async def test_r2_score(
        regression_labels_real: Vector[float],
        regression_labels_pred: Vector[float],
    ) -> None:
        """
        Test for validating the accuracy of the secure r2_score
        implementation versus sklearn.

        :param regression_labels_real: Real labels
        :param regression_labels_pred: Predicted labels
        """
        secure_regression_labels_real = mpyc_input(regression_labels_real, SECFXP)
        secure_regression_labels_pred = mpyc_input(regression_labels_pred, SECFXP)

        plain_r2_score = plaintext_metrics.r2_score(
            regression_labels_real, regression_labels_pred
        )
        secure_r2_score = metrics.r2_score(
            secure_regression_labels_real, secure_regression_labels_pred
        )
        result = await mpyc_output(secure_r2_score)
        assert result == pytest.approx(plain_r2_score, abs=TOLERABLE_ABS_ERROR)

    @staticmethod
    async def test_adj_r2_score(
        regression_labels_real: Vector[float],
        regression_labels_pred: Vector[float],
    ) -> None:
        """
        Test for validating the accuracy of the secure adj_r2_score
        implementation versus sklearn.

        :param regression_labels_real: Real labels
        :param regression_labels_pred: Predicted labels
        """
        secure_regression_labels_real = mpyc_input(regression_labels_real, SECFXP)
        secure_regression_labels_pred = mpyc_input(regression_labels_pred, SECFXP)

        plain_adj_r2_score = plaintext_metrics.r2_score(
            regression_labels_real, regression_labels_pred
        )
        secure_adj_r2_score = metrics.r2_score(
            secure_regression_labels_real, secure_regression_labels_pred
        )
        result = await mpyc_output(secure_adj_r2_score)
        assert result == pytest.approx(plain_adj_r2_score, abs=TOLERABLE_ABS_ERROR)
