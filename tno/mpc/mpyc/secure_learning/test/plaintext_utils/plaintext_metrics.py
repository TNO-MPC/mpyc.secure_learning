"""
Metrics for using classification and regression on plaintext
"""
from typing import Union

import numpy as np
import sklearn.metrics

from tno.mpc.mpyc.secure_learning.utils import NumpyNumberArray, NumpyOrVector


# Classification metrics
def accuracy_score(y_real: NumpyOrVector, y_pred: NumpyOrVector) -> float:
    """
    Computes the accuracy of the predicted labels. Accuracy is computed as
    the ratio of all correctly predicted labels over the number of predicted
    labels.

    :param y_real: Real labels
    :param y_pred: Predicted labels
    :return: Accuracy of given predictions
    """
    return sklearn.metrics.accuracy_score(y_real, y_pred)


def precision_score(
    y_real: NumpyOrVector, y_pred: NumpyOrVector, pos_label: Union[str, int]
) -> float:
    """
    Computes the precision of the predicted labels of category
    pos_label. Precision is computed as the ratio of all correctly
    predicted pos_label over the number of predicted pos_label. This is
    an indication how precise the predictions of pos_label are: given
    prediction pos_label, how likely is it that the true label is
    pos_label.

    :param y_real: Real labels
    :param y_pred: Predicted labels
    :param pos_label: Label to compute precision of
    :return: Precision of given predictions
    """
    return sklearn.metrics.precision_score(y_real, y_pred, pos_label=pos_label)


def recall_score(
    y_real: NumpyOrVector, y_pred: NumpyOrVector, pos_label: Union[str, int]
) -> float:
    """
    Computes the recall of the predicted labels of category pos_label.
    Recall is computed as the ratio of all correctly predicted pos_label
    over the number of real pos_label. This is an indication how many
    actual pos_label we misclassified.

    :param y_real: Real labels
    :param y_pred: Predicted labels
    :param pos_label: Label to compute recall of
    :return: Recall of given predictions
    """
    return sklearn.metrics.recall_score(y_real, y_pred, pos_label=pos_label)


def f1_score(
    y_real: NumpyOrVector, y_pred: NumpyOrVector, pos_label: Union[str, int]
) -> float:
    """
    F1-score for given predicted and real target labels.

    :param y_real: Real labels
    :param y_pred: Predicted labels
    :param pos_label: Label to compute f1 for
    :return: f1 score
    """
    return sklearn.metrics.f1_score(y_real, y_pred, pos_label=pos_label)


# Regression metrics
def r2_score(y_real: NumpyOrVector, y_pred: NumpyOrVector) -> float:
    """
    R-squared value for given predicted and real target values.

    :param y_real: Real target values
    :param y_pred: Predicted target values
    :return: R-squared value
    """
    return sklearn.metrics.r2_score(y_real, y_pred)


def adj_r2_score(
    y_real: NumpyOrVector, y_pred: NumpyOrVector, n_features: int
) -> float:
    """
    Adjusted R-squared value for given predicted and real target values.

    :param y_real: Real target values
    :param y_pred: Predicted target values
    :param n_features: Number of features
    :return: Adjusted R-squared value
    """
    n_samples = len(y_pred)
    r2 = r2_score(y_real, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def mean_squared_error(y_real: NumpyOrVector, y_pred: NumpyOrVector) -> float:
    """
    Compute residual mean of squares. Residual mean of squares equals the
    mean of squares of deviations between predicted and real values.

    :param y_real: Real target values
    :param y_pred: Predicted target values
    :return: Residual mean of squares
    """
    return sklearn.metrics.mean_squared_error(y_real, y_pred)


def mean_squared_model(y_real: NumpyOrVector) -> float:
    """
    Compute explained mean of squares. Explained mean of squares equals the
    mean of squares of deviations from the mean.

    :param y_real: Input data
    :return: Explained mean of squares
    """
    y: NumpyNumberArray = np.asarray(y_real)
    dev_from_mean = y - y.mean()
    return float(np.inner(dev_from_mean, dev_from_mean)) / len(y)
