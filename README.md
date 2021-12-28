# TNO MPC Lab - MPyC - Secure Learning

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.mpyc.secure_learning is part of the TNO Python Toolbox.

This library has been developed with funding from different projects.

In particular, the basic building blocks and an initial version of this library have been developed within the VP AI program (2018) and the ERP AI program (2019), including an SVM model and initial versions of other models.

The current secure logistic regression model has been developed within the TKI HTSM LANCELOT project, a research collaboration between [TNO](https://www.tno.nl/en/), [IKNL](https://iknl.nl/) and [Janssen](http://www.janssen.com/).

LANCELOT is partly funded by PPS-surcharge for Research and Innovation of the Dutch Ministry of Economic Affairs and Climate Policy.

The secure lasso regression model has been developed in the BigMedilytics project. This project has received funding from the European Union’s Horizon 2020 research and innovation program under Grant Agreement No. 780495.

In collaboration with the MPC Lab, BigMedilytics, LANCELOT, [NLAIC](https://nlaic.com/en/about-nl-aic/) and [Appl.AI](https://www.tno.nl/en/focus-areas/artificial-intelligence/), contributed to a restructuring of the codebase to ensure a generic reusable library which can be expanded with other models and functionalities.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*  
*This implementation of cryptographic software has not been audited. Use at your own risk.*

## Content Explanation

Implementation based on Secure Multi-Party Computation (MPC) for training and evaluating of several machine learning models.
It makes use of the [MPyC](https://pypi.org/project/mpyc/) framework.

## Features

The library implements secure versions of popular machine learning methods in the form of MPC protocols.
The underlying MPC functionalities are provided by the MPyC framework.

The library contains both regression and classification algorithms.

In particular, linear regression is implemented, with l2-penalty (Ridge), l1-penalty (Lasso), or a combination of both (ElasticNets).
For what concerns classification problems, Support Vector Machines (SVM) are implemented, as well as logistic "regression".
For the latter, the user can choose between an accurate implementation of the logistic function, and an approximated, but faster version.
l1 and/or l2 penalties can also be selected.

The library allows users to choose either the gradient-descent or the SAG solver in order to train the implemented models.

## Limitations

Currently, no code is provided to securely apply the trained models.

## Documentation

Documentation of the tno.mpc.mpyc.secure_learning package can be found [here](https://docs.mpc.tno.nl/mpyc/secure_learning/0.2.3).

## Install

Easily install the tno.mpc.mpyc.secure_learning package using pip:
```console
$ python -m pip install tno.mpc.mpyc.secure_learning
```

### Note:
A significant performance improvement can be achieved by installing the GMPY2 library.
```console
$ python -m pip install 'tno.mpc.mpyc.secure_learning[gmpy]'
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.mpc.mpyc.secure_learning[tests]'
```

## Usage

Run these examples as `python example.py --no-log` to suppress the MPyC barrier logging. Append the argument `-M 3` to simulate a three-party protocol.

### Example Usage

<details>
<summary><b>Click here for an example of securely training a simple linear regression model with L2 penalty (Ridge).</b></summary>

> `example.py`
> ```python
> import numpy as np
> from mpyc.runtime import mpc
> from sklearn import datasets
> from sklearn.linear_model import Ridge as RidgeSK
> 
> import tno.mpc.mpyc.secure_learning.test.plaintext_utils.plaintext_objective_functions as plain_obj
> from tno.mpc.mpyc.secure_learning import (
>     PenaltyTypes,
>     Ridge,
>     SolverTypes,
> )
> 
> # Notice that we use the entire dataset to train the model
> n_samples = 50
> n_features = 5
> # Fixed random state for reproducibility
> random_state = 3
> tolerance = 1e-4
> 
> secnum = mpc.SecFxp(l=64, f=32)
> 
> 
> def get_mpc_data(X, y):
>     X_mpc = [[secnum(x, integral=False) for x in row] for row in X.tolist()]
>     y_mpc = [secnum(y, integral=False) for y in y.tolist()]
>     return X_mpc, y_mpc
> 
> 
> def distribute_data_over_players(X_mpc, y_mpc):
>     X_shared = [mpc.input(row, senders=0) for row in X_mpc]
>     y_shared = mpc.input(y_mpc, senders=0)
>     return X_shared, y_shared
> 
> 
> async def ridge_regression_example():
>     print("Ridge regression with gradient descent method")
>     alpha = 0.2
> 
>     # Create regression dataset
>     X, y = datasets.make_regression(
>         n_samples=n_samples,
>         n_features=n_features,
>         noise=25.0,
>         random_state=random_state,
>     )
>     X = np.array(X)
>     y = np.array(y)
>     X_mpc, y_mpc = get_mpc_data(X, y)
> 
>     async with mpc:
>         X_shared, y_shared = distribute_data_over_players(X_mpc, y_mpc)
> 
>     # Train secure model
>     model = Ridge(solver_type=SolverTypes.GD, alpha=alpha)
>     async with mpc:
>         weights = await model.compute_weights_mpc(
>             X_shared,
>             y_shared,
>             tolerance=tolerance,
>         )
> 
>     # Results of secure model
>     objective = plain_obj.objective(X, y, weights, "linear", PenaltyTypes.L2, alpha)
>     print("Securely obtained coefficients:", weights)
>     print("* objective:", objective)
> 
>     # Train plaintext model
>     model_sk = RidgeSK(
>         alpha=len(X) * alpha,
>         solver="saga",
>         random_state=random_state,
>         fit_intercept=True,
>     )
>     model_sk.fit(X, y)
> 
>     # Results of plaintext model
>     weights_sk = np.append([model_sk.intercept_], model_sk.coef_).tolist()
>     objective_sk = plain_obj.objective(
>         X, y, weights_sk, "linear", PenaltyTypes.L2, alpha
>     )
>     print("Sklearn obtained coefficients: ", weights_sk)
>     print("* objective:", objective_sk)
> 
> 
> if __name__ == "__main__":
>     mpc.run(ridge_regression_example())
> ```

</details>

<details>
<summary><b>Click here for an example of securely training a logistic regression model with L1 penalty.</b></summary>

> `example.py`
> ```python
> import numpy as np
> from mpyc.runtime import mpc
> from sklearn import datasets
> from sklearn.linear_model import LogisticRegression as LogisticRegressionSK
> 
> import tno.mpc.mpyc.secure_learning.test.plaintext_utils.plaintext_objective_functions as plain_obj
> from tno.mpc.mpyc.secure_learning import (
>     ExponentiationTypes,
>     Logistic,
>     PenaltyTypes,
>     SolverTypes,
> )
> 
> # Notice that we use the entire dataset to train the model
> n_samples = 50
> n_features = 5
> # Fixed random state for reproducibility
> random_state = 3
> tolerance = 1e-4
> 
> secnum = mpc.SecFxp(l=64, f=32)
> 
> 
> def get_mpc_data(X, y):
>     X_mpc = [[secnum(x, integral=False) for x in row] for row in X.tolist()]
>     y_mpc = [secnum(y, integral=False) for y in y.tolist()]
>     return X_mpc, y_mpc
> 
> 
> def distribute_data_over_players(X_mpc, y_mpc):
>     X_shared = [mpc.input(row, senders=0) for row in X_mpc]
>     y_shared = mpc.input(y_mpc, senders=0)
>     return X_shared, y_shared
> 
> 
> async def logistic_regression_example():
>     print(
>         "Classification (Logistic regression) with l1 penalty, with gradient descent method"
>     )
>     alpha = 0.1
> 
>     # Create classification dataset
>     X, y = datasets.make_classification(
>         n_samples=n_samples,
>         n_features=n_features,
>         n_informative=1,
>         n_redundant=0,
>         n_classes=2,
>         n_clusters_per_class=1,
>         random_state=random_state,
>         shift=0,
>     )
>     # Transform labels from {0, 1} to {-1, +1}.
>     y = [-1 if x == 0 else 1 for x in y]
>     X = np.array(X)
>     y = np.array(y)
>     X_mpc, y_mpc = get_mpc_data(X, y)
> 
>     async with mpc:
>         X_shared, y_shared = distribute_data_over_players(X_mpc, y_mpc)
> 
>     # Train secure model with approximation of logistic function (faster, less accurate)
>     model = Logistic(
>         solver_type=SolverTypes.GD,
>         exponentiation=ExponentiationTypes.APPROX,
>         penalty=PenaltyTypes.L1,
>         alpha=alpha,
>     )
>     async with mpc:
>         weights_approx = await model.compute_weights_mpc(
>             X_shared, y_shared, tolerance=tolerance
>         )
> 
>     # Results of secure model (approximated logistic function)
>     objective_approx = plain_obj.objective(
>         X, y, weights_approx, "logistic", PenaltyTypes.L1, alpha
>     )
>     print(
>         "Securely obtained coefficients (approximated exponentiation):",
>         weights_approx,
>     )
>     print("* objective:", objective_approx)
> 
>     # Train secure model with exact logistic function (slower, more accurate)
>     model = Logistic(
>         solver_type=SolverTypes.GD,
>         exponentiation=ExponentiationTypes.EXACT,
>         penalty=PenaltyTypes.L1,
>         alpha=alpha,
>     )
>     async with mpc:
>         weights_exact = await model.compute_weights_mpc(
>             X_shared, y_shared, tolerance=tolerance
>         )
> 
>     # Results of secure model (exact logistic function)
>     objective_exact = plain_obj.objective(
>         X, y, weights_exact, "logistic", PenaltyTypes.L1, alpha
>     )
>     print(
>         "Securely obtained coefficients (exact exponentiation):       ",
>         weights_exact,
>     )
>     print("* objective:", objective_exact)
> 
>     # Train plaintext model
>     model_sk = LogisticRegressionSK(
>         solver="saga",
>         random_state=random_state,
>         fit_intercept=True,
>         penalty="l1",
>         C=1 / (len(X) * alpha),
>     )
>     model_sk.fit(X, y)
>     weights_sk = np.append([model_sk.intercept_], model_sk.coef_).tolist()
> 
>     # Results of plaintest model
>     objective_sk = plain_obj.objective(
>         X, y, weights_sk, "logistic", PenaltyTypes.L1, alpha
>     )
>     print("Sklearn obtained coefficients:                               ", weights_sk)
>     print("* objective:", objective_sk)
> 
> 
> if __name__ == "__main__":
>     mpc.run(logistic_regression_example())
> ```

</details>
