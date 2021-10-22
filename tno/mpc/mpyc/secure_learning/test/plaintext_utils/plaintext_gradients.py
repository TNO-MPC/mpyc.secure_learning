"""
Gradient functions on plaintext data
"""
import math

import numpy as np

from tno.mpc.mpyc.secure_learning.utils import (
    NumpyNumberArray,
    NumpyOrMatrix,
    NumpyOrVector,
)


def plain_linear_gradient(
    X: NumpyOrMatrix,
    y: NumpyOrVector,
    weights: NumpyOrVector,
    grad_per_sample: bool,
) -> NumpyNumberArray:
    r"""
    Gradient for Linear regression. Optimizes a model with objective
    function:
    $$\frac{1}{(2n_{\textrm{samples}}} \times ||y - X_times_w||^2_2.$$

    The gradient is given by
    $$g(X, y, w) = \\frac{1}{{n}_{\textrm{samples}}} \times X^T \times (X_times_w - y).$$

    :param X: Independent variables.
    :param y: Dependent variables.
    :param weights: Current weights vector.
    :param grad_per_sample: Return a 2D-array with gradient per sample
        instead of aggregated (summed) 1D-gradient.

    :return: Gradient for linear regression as specified in above
        docstring, evaluated from the provided parameters.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    resulting_array: NumpyNumberArray
    if not grad_per_sample:
        resulting_array = 1 / X.shape[0] * np.dot(X.T, (np.dot(X, weights) - y))  # type: ignore[no-untyped-call]
    else:
        resulting_array = (1 / X.shape[0]) * np.asarray(
            [
                plain_linear_gradient(
                    X[[i], :], y_sample, weights, grad_per_sample=False
                )
                for (i, y_sample) in enumerate(y)
            ]
        )
    return resulting_array


def transform_predicted_label(x: float) -> float:
    """
    Transform labels that assume values in $[0, 1]$ to labels that assume
    values in $[-1, 1]$.

    :param x: Label prediction that assumes values in $[0, 1]$.
    :return: Label prediction that assumes values in $[-1, 1]$.
    """
    return 2 * x - 1


def logit(x: float) -> float:
    r"""
    Evaluate the logistic function $\frac{1}{1 + e^x}$.

    :param x: Variable in the above logistic function.
    :return: Evaluation of the above logistic function.
    """

    if x >= 0:
        return math.exp(-x) / (math.exp(-x) + 1)
    return 1 / (1 + math.exp(x))


def plain_exact_logistic_gradient(
    X: NumpyOrMatrix,
    y: NumpyOrVector,
    weights: NumpyOrVector,
    grad_per_sample: bool,
) -> NumpyNumberArray:
    r"""
    Gradient for Logistic regression. Optimizes a model with objective
    function:
    $$\left(\frac{1}{2n_{\textrm{samples}}}\right) \sum_{i=1}^{n_{\textrm{samples}}}\left(\textrm{-}(1+y_i) \log(h_w(x_i)) - (1-y_i) \log(1-h_w(x_i))\right).$$

    Here,
    $$h_w(x) = \frac{1}{(1 + e^{-w^T x}}.$$

    Labels $y_i$ are assumed to have value $-1$ or $1$.

    The gradient is given by:
    $$g(X, y, w) = \left(\frac{1}{2} \times {n}_{\textrm{samples}}\right) \sum_{i=1}^{{n}_\textrm{samples}} x_i^T \left( (2h_w(x_i) - 1) - y \right).$$

    :param X: Independent variables.
    :param y: Dependent variables.
    :param weights: Current weights vector.
    :param grad_per_sample: Return a 2D-array with gradient per sample
        instead of aggregated (summed) 1D-gradient.
    :return: Gradient for logistic regression as specified in above
        docstring, evaluated from the provided parameters.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    resulting_array: NumpyNumberArray
    if not grad_per_sample:
        resulting_array = (1 / X.shape[0]) * np.sum(
            [
                x_sample
                * (
                    transform_predicted_label(logit(-np.dot(x_sample, weights)))  # type: ignore[no-untyped-call]
                    - y_sample
                )
                / 2
                for (x_sample, y_sample) in zip(X, y)
            ],
            axis=0,
        )
    else:
        resulting_array = (1 / X.shape[0]) * np.asarray(
            [
                plain_exact_logistic_gradient(
                    X[[i], :], [y_sample], weights, grad_per_sample=False
                )
                for (i, y_sample) in enumerate(y)
            ]
        )
    return resulting_array


def approx_logit(x: float) -> float:
    r"""
    Approximate the logistic function $$\frac{1}{1 + e^x}$$.

    :param x: Variable in the above logistic function.
    :return: Approximation of the above logistic function.
    """

    if x < -0.5:
        return 0
    if x > 0.5:
        return 1
    return x + 0.5


def plain_approximate_logistic_gradient(
    X: NumpyOrMatrix,
    y: NumpyOrVector,
    weights: NumpyOrVector,
    grad_per_sample: bool,
) -> NumpyNumberArray:
    r"""
    Approximate gradient for Logistic regression. Optimizes a model with
    objective function:
    $$\left(\frac{1}{2{n}_{\textrm{samples}}}\right) \sum_{i=1}^{{n}_{\textrm{samples}}}\left(\textrm{-}(1+y_i) \log(h_w(x_i)) - (1-y_i) \log(1-h_w(x_i))\right).$$

    Here,
    $$h_w(x) = \frac{1}{(1 + e^{-w^T x}}.$$

    Labels $y_i$ are assumed to have value $-1$ or $1$.

    The gradient is given by:
    $$g(X, y, w) = \left(\frac{1}{2} \times {n}_{\textrm{samples}}\right) \sum_{i=1}^{{n}_\textrm{samples}} x_i^T \left( (2h_w(x_i) - 1) - y \right).$$

    :param X: Independent variables.
    :param y: Dependent variables.
    :param weights: Current weights vector.
    :param grad_per_sample: Return a 2D-array with gradient per sample
        instead of aggregated (summed) 1D-gradient.
    :return: Gradient for logistic regression as specified in above
        docstring, evaluated from the provided parameters.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    resulting_array: NumpyNumberArray
    if not grad_per_sample:
        resulting_array = (1 / X.shape[0]) * np.sum(
            [
                x_sample
                * (
                    transform_predicted_label(approx_logit(np.dot(x_sample, weights)))  # type: ignore[no-untyped-call]
                    - y_sample
                )
                / 2
                for (x_sample, y_sample) in zip(X, y)
            ],
            axis=0,
        )
    else:
        resulting_array = (1 / X.shape[0]) * np.asarray(
            [
                plain_approximate_logistic_gradient(
                    X[[i], :], [y_sample], weights, grad_per_sample=False
                )
                for (i, y_sample) in enumerate(y)
            ]
        )
    return resulting_array


def hinge_loss_grad(
    data: NumpyNumberArray, weights: NumpyNumberArray, true_label: float
) -> NumpyNumberArray:
    """
    Compute the (degenerate) gradient of the hinge loss.

    :param data: Samples of independent variables.
    :param weights: Current weights vector.
    :param true_label: True label of data samples.
    :return: (Degenerate) gradient of the hinge loss.
    """
    pred = np.dot(data, weights)  # type: ignore[no-untyped-call]
    if true_label * pred >= 1:
        return np.zeros_like(weights)
    return -true_label * data


def plain_svm_gradient(
    X: NumpyOrMatrix,
    y: NumpyOrVector,
    weights: NumpyOrVector,
    grad_per_sample: bool,
) -> NumpyNumberArray:
    r"""
    Approximate Gradient for Logistic regression. Optimizes a model with
    objective function
    $$\frac{1}{{n}_{\textrm{samples}}} \sum_{i=1}^{{n}_{\textrm{samples}}} h_w(x_i, y_i).$$

    Here, $h_w(x, y)$ is the hinge loss, defined as:
    $$\max(0, 1 - y \times (w^T x)).$$

    Labels $y_i$ are assumed to obtain values $-1$ and $1$.

    :param X: Independent variables.
    :param y: Dependent variables.
    :param weights: Current weights vector.
    :param grad_per_sample: Return a 2D-array with gradient per sample
        instead of aggregated (summed) 1D-gradient.
    :return: Approximate gradient for logistic regression as specified in
        above docstring, evaluated from the provided parameters.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    resulting_array: NumpyNumberArray
    if not grad_per_sample:
        resulting_array = (1 / X.shape[0]) * np.sum(
            [
                hinge_loss_grad(x_sample, weights, y_sample)
                for (x_sample, y_sample) in zip(X, y)
            ],
            axis=0,
        )
    else:
        resulting_array = (1 / X.shape[0]) * np.asarray(
            [
                plain_svm_gradient(
                    X[[i], :], [y_sample], weights, grad_per_sample=False
                )
                for (i, y_sample) in enumerate(y)
            ]
        )
    return resulting_array
