"""
Provides classification and regression metrics.
"""

import numpy as np

from tno.mpc.mpyc.secure_learning.exceptions import SecureLearnValueError
from tno.mpc.mpyc.secure_learning.utils import NumpyObjectArray, SecNumTypesTV, Vector


# Classification metrics
def accuracy_score(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
) -> SecNumTypesTV:
    """
    Computes the accuracy of the predicted labels. Accuracy is computed as
    the ratio of all correctly predicted labels over the number of predicted
    labels.

    :param y_real: Real labels (-/+ 1)
    :param y_pred: Predicted labels (-/+ 1)
    :return: Accuracy of given predictions
    """
    # -1 if TN, 1 if TP, 0 if FN or FP
    signed_pred_correct = (np.asarray(y_pred) + np.asarray(y_real)) / 2
    accuracy: SecNumTypesTV = np.inner(signed_pred_correct, signed_pred_correct) / len(  # type: ignore[no-untyped-call]
        y_real
    )
    return accuracy


def precision_score(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
    pos_label: int,
) -> SecNumTypesTV:
    """
    Computes the precision of the predicted labels of category
    pos_label. Precision is computed as the ratio of all correctly
    predicted pos_label over the number of predicted pos_label. This is
    an indication how precise the predictions of pos_label are: given
    prediction pos_label, how likely is it that the true label is
    pos_label.

    :param y_real: Real labels (-/+ 1)
    :param y_pred: Predicted labels (-/+ 1)
    :param pos_label: Label (value) to compute precision of
    :raise SecureLearnValueError: pos_label value must be either -1 or 1
    :return: Precision of given predictions
    """
    if pos_label not in [-1, 1]:
        raise SecureLearnValueError(
            f"Expected pos_label in [-1, 1], but received {pos_label}."
        )
    y_real_np = np.asarray(y_real)
    y_pred_np = np.asarray(y_pred)

    # Reduce to case target label = 1
    if pos_label == -1:
        y_real_np = -y_real_np
        y_pred_np = -y_pred_np

    # 1 if true target, 0 else
    num_true_targets: SecNumTypesTV = np.sum(
        (y_real_np * y_pred_np + y_real_np + y_pred_np + 1) / 4
    )
    num_pred_targets: SecNumTypesTV = np.sum((y_pred_np + 1) / 2)
    return num_true_targets / num_pred_targets


def recall_score(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
    pos_label: int,
) -> SecNumTypesTV:
    """
    Computes the recall of the predicted labels of category pos_label.
    Recall is computed as the ratio of all correctly predicted pos_label
    over the number of real pos_label. This is an indication how many
    actual pos_label we misclassified.

    :param y_real: Real labels (-/+ 1)
    :param y_pred: Predicted labels (-/+ 1)
    :param pos_label: Label (value) to compute recall of
    :raise SecureLearnValueError: pos_label value must be either -1 or 1
    :return: Recall of given predictions
    """
    if pos_label not in [-1, 1]:
        raise SecureLearnValueError(
            f"Expected pos_label in [-1, 1], but received {pos_label}."
        )
    y_real_np = np.asarray(y_real)
    y_pred_np = np.asarray(y_pred)

    # Reduce to case target label = 1
    if pos_label == -1:
        y_real_np = -y_real_np
        y_pred_np = -y_pred_np

    # 1 if true target, 0 else
    num_true_targets: SecNumTypesTV = np.sum(
        (y_real_np * y_pred_np + y_real_np + y_pred_np + 1) / 4
    )
    num_real_targets: SecNumTypesTV = np.sum((y_real_np + 1) / 2)
    return num_true_targets / num_real_targets


def f1_score(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
    pos_label: int,
) -> SecNumTypesTV:
    """
    F1-score for given predicted and real target labels.

    :param y_real: Real labels (-/+ 1)
    :param y_pred: Predicted labels (-/+ 1)
    :param pos_label: Label to compute f1 score of
    :raise SecureLearnValueError: pos_label value must be either -1 or 1
    :return: F1 score
    """
    if pos_label not in [-1, 1]:
        raise SecureLearnValueError(
            f"Expected pos_label in [-1, 1], but received {pos_label}."
        )
    precision: SecNumTypesTV = precision_score(y_real, y_pred, pos_label)
    recall: SecNumTypesTV = recall_score(y_real, y_pred, pos_label)
    return 2 * (precision * recall) / (precision + recall)


# Regression metrics
def mean_squared_error(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
) -> SecNumTypesTV:
    """
    Compute residual mean of squares. Residual mean of squares
    equals the mean of squares of deviations between predicted
    and real values.

    :param y_real: Real labels (-/+ 1)
    :param y_pred: Predicted labels (-/+ 1)
    :return: Residual mean of squares
    """
    pred_error: NumpyObjectArray = np.asarray(y_pred) - np.asarray(y_real)
    res_mean_sq: SecNumTypesTV = np.inner(pred_error, pred_error) / len(y_real)  # type: ignore[no-untyped-call]
    return res_mean_sq


def mean_squared_model(y_real: Vector[SecNumTypesTV]) -> SecNumTypesTV:
    """
    Compute explained mean of squares. Explained sum of
    squares equals the mean of squares of deviations from the
    mean.

    :param y_real: Input
    :return: Explained mean of squares
    """
    y = np.asarray(y_real)
    dev_from_mean = y - y.sum() / len(y)
    explained_mean_sq: SecNumTypesTV = np.inner(dev_from_mean, dev_from_mean) / len(y)  # type: ignore[no-untyped-call]
    return explained_mean_sq


def r2_score(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
) -> SecNumTypesTV:
    """
    R-squared value for given predicted and real target values.

    :param y_real: Real target values
    :param y_pred: Predicted target values
    :return: R-squared value
    """
    mean_explained_ss: SecNumTypesTV = mean_squared_model(y_real)
    mean_residual_ss: SecNumTypesTV = mean_squared_error(y_real, y_pred)
    return 1 - (mean_residual_ss / mean_explained_ss)


def adj_r2_score(
    y_real: Vector[SecNumTypesTV],
    y_pred: Vector[SecNumTypesTV],
    n_features: int,
) -> SecNumTypesTV:
    """
    Adjusted R-squared value for given predicted
    and real target values.

    :param y_real: Real target values
    :param y_pred: Predicted target values
    :param n_features: Number of features
    :return: Adjusted R-squared value
    """
    n_samples = len(y_pred)
    r_squared = r2_score(y_real, y_pred)
    return 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
