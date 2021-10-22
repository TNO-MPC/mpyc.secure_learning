"""
Objective functions using plaintext
"""
import math
from typing import List, cast

import numpy as np

from tno.mpc.mpyc.secure_learning.exceptions import SecureLearnValueError
from tno.mpc.mpyc.secure_learning.models import PenaltyTypes
from tno.mpc.mpyc.secure_learning.utils import (
    NumpyNumberArray,
    NumpyOrMatrix,
    NumpyOrVector,
)


def regression_prediction(X: NumpyOrMatrix, weights: NumpyOrVector) -> NumpyNumberArray:
    """
    Predicts target values for input data to regression model.

    :param X: Input data with all features
    :param weights: Weight vector of regression model
    :return: Target values of regression model
    """

    if len(X[0]) == len(weights):
        return cast(NumpyNumberArray, np.matmul(X, weights))
    return cast(NumpyNumberArray, np.matmul(X, weights[1:]) + weights[0])


def logistic_classification_prediction(
    X: NumpyOrMatrix, weights: NumpyOrVector, prob: float = 0.5
) -> NumpyNumberArray:
    """
    Predicts labels for input data to logistic regression classification
    model. Label -1 is assigned if the predicted probability is less then
    `prob`, otherwise label +1 is assigned.

    :param X: Input data with all features
    :param weights: Weight vector of classification model
    :param prob: Threshold for labelling
    :return: Target labels of logistic regression classification model
    """
    pred_probs = classification_probabilities(X, weights)
    vfunc = np.vectorize(lambda pred_prob, prob: 1 if pred_prob >= prob else -1)
    return cast(NumpyNumberArray, vfunc(pred_probs, prob))


def SVM_classification_prediction(
    X: NumpyOrMatrix, weights: NumpyOrVector
) -> List[float]:
    """
    Predicts labels for input data to SVM classification model. Label -1 is
    assigned if the inner product of the features and the weight vector
    (corrected with intercept) is negative, otherwise label +1 is assigned.

    :param X: Input data with all features
    :param weights: Weight vector of classification model
    :return: Target labels of SVM classification model
    """
    X_times_w = np.matmul(X, weights[1:])
    return [1 if X_times_wi + weights[0] >= 0 else -1 for X_times_wi in X_times_w]


def classification_probabilities(
    X: NumpyOrMatrix, weights: NumpyOrVector
) -> NumpyOrVector:
    """
    Computes probability for label +1 from input data and classification
    model weight vector.

    :param X: Input data with all features
    :param weights: Weight vector of classification model
    :return: Predicted probabilities for label +1
    """
    if len(X[0]) != len(weights):
        X = np.concatenate((np.expand_dims([1] * len(X), axis=1), X), axis=1)  # type: ignore[no-untyped-call]
    inners = np.dot(X, weights)  # type: ignore[no-untyped-call]
    vfunc = np.vectorize(standard_logistic_function)
    return cast(NumpyNumberArray, vfunc(inners))


def standard_logistic_function(x: float) -> float:
    """
    Computes value of standard logistic function.

    :param x: Point to be evaluated
    :return: Value of logistic function evaluated in x
    """
    return 1 / (1 + math.exp(-x))


def objective(
    X: NumpyOrMatrix,
    y: NumpyOrVector,
    weights: NumpyOrVector,
    model: str,
    penalty: PenaltyTypes,
    alpha: float,
) -> float:
    """
    Objective value of regression or classification model.

    :param X: Input data with all features
    :param y: Target variable or label
    :param weights: Weight vector of model
    :param model: Name of the model that was trained
    :param penalty: Type of penalty that is applied
    :param alpha: Regularization parameter
    :raise NotImplementedError: If no objective function has been implemented
        for the given model

    :return: Objective value for given model, model weight vector and data.
    """
    if model == "lasso":
        penalty = PenaltyTypes.L1
        objective_value = linear_objective(X, y, weights)
    elif model == "linear":
        objective_value = linear_objective(X, y, weights)
    elif model in ("logistic", "logistic_approx"):
        objective_value = logistic_objective(X, y, weights)
    elif model == "ridge":
        penalty = PenaltyTypes.L2
        objective_value = linear_objective(X, y, weights)
    elif model == "SVM":
        objective_value = SVM_objective(X, y, weights)
    else:
        raise NotImplementedError(
            f"No objective function is implemented for model {model}."
        )

    if penalty == PenaltyTypes.ELASTICNET:
        raise NotImplementedError("Still need to take care of passing all parameters")

    # Assuming intercept for now
    if penalty in (PenaltyTypes.L1,):
        objective_value += alpha * l1_penalty(weights[1:])
    if penalty in (PenaltyTypes.L2,):
        objective_value += alpha * l2_penalty(weights[1:])

    return objective_value


def l1_penalty(weights: NumpyOrVector) -> float:
    """
    Compute L1 penalty of a vector.

    :param weights: Vector of interest.
    :return: L1 penalty of vector.
    """
    return sum([abs(x) for x in weights])


def l2_penalty(weights: NumpyOrVector) -> float:
    """
    Compute L2 penalty of a vector.

    :param weights: Vector of interest.
    :return: L2 penalty of vector.
    """
    return float(np.inner(weights, weights) / 2)  # type: ignore[no-untyped-call]


def linear_objective(
    X: NumpyOrMatrix, y: NumpyOrVector, weights: NumpyOrVector
) -> float:
    """
    Objective value of linear regression model.

    :param X: Input data with all features
    :param y: Target variable or label
    :param weights: Weight vector of model
    :return: LinearObjective value for given model weight vector and data
    """
    error: NumpyNumberArray = regression_prediction(X, weights) - y
    error_squared: float = np.inner(error, error)  # type: ignore[no-untyped-call]
    return error_squared / (2 * len(y))


def logistic_objective(
    X: NumpyOrMatrix, y: NumpyOrVector, weights: NumpyOrVector
) -> float:
    """
    Objective value of logistic regression model.

    :param X: Input data with all features
    :param y: Target variable or label
    :param weights: Weight vector of model
    :return: Objective value for given model weight vector and data
    """
    prob_label_1_array = classification_probabilities(X, weights)

    def misclassification_prob(prob_label_1: int, real_label: int) -> float:
        """
        Computes the probability of a misclassification, given the actual value of the label and the probability that the model returns that value.

        :param prob_label_1: Probability that the model returns label 1
        :param real_label: Actual label, assumed to be either 1 or -1
        :raise SecureLearnValueError: if the received label is not 1 nor -1
        :return: The probability of a misclassification
        """
        if real_label == -1:
            return math.log(1 - prob_label_1)
        if real_label == 1:
            return math.log(prob_label_1)
        raise SecureLearnValueError(
            f"Expected to receive label -1 or +1, received {real_label}"
        )

    vfunc = np.vectorize(misclassification_prob)
    misclassification_probs = vfunc(prob_label_1_array, y)
    return float(-1 / len(y) * misclassification_probs.sum())


def SVM_objective(X: NumpyOrMatrix, y: NumpyOrVector, weights: NumpyOrVector) -> float:
    """
    Objective value of SVM model with hinge loss function.

    :param X: Input data with all features
    :param y: Target variable or label
    :param weights: Weight vector of model
    :return: Objective value for given model weight vector and data
    """
    X_times_w = np.matmul(X, weights[1:])
    error = [
        hinge_loss(yi, X_times_wi + weights[0])
        for (yi, X_times_wi) in zip(y, X_times_w)
    ]
    return sum(error) / len(X)


def hinge_loss(predicted_value: float, real_label: float) -> float:
    """
    Computes hinge loss for given predicted value, given the real label.

    :param predicted_value: inner product of data sample and weight
            vector, possibly corrected by intercept
    :param real_label: Real labbel
    :return: Hinge loss of datapoint
    """
    return max(0, 1 - real_label * predicted_value)
