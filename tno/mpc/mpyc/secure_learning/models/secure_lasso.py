"""
Implementation of LASSO regression model.
"""
from tno.mpc.mpyc.secure_learning.models.secure_linear import Linear
from tno.mpc.mpyc.secure_learning.models.secure_model import PenaltyTypes, SolverTypes


class Lasso(Linear):
    r"""
    Solver for LASSO regression. Optimizes a model with objective function
    $$\frac{1}{2n_{\textrm{samples}}} \times ||y - X_times_w||^2_2 + \alpha_1 ||w||_1$$
    """
    name = "LASSO regression"

    def __init__(
        self,
        solver_type: SolverTypes = SolverTypes.GD,
        alpha: float = 1,
    ) -> None:
        """
        Constructor method.

        :param solver_type: Solver type to use (e.g. Gradient Descent aka GD)
        :param alpha: Regularization parameter
        """
        super().__init__(solver_type, penalty=PenaltyTypes.L1, alpha=alpha)
