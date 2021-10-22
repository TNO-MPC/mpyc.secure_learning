"""
Implementation of Ridge regression model.
"""
from tno.mpc.mpyc.secure_learning.models.secure_linear import Linear
from tno.mpc.mpyc.secure_learning.models.secure_model import PenaltyTypes, SolverTypes


class Ridge(Linear):
    r"""
    Solver for Ridge regression. Optimizes a model with objective function
    $$\frac{1}{{2n}_{\textrm{samples}}} \times ||y - Xw||^2_2 +  \frac{\alpha}{2} \times ||w||^2_2$$
    """
    name = "Ridge regression"

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
        super().__init__(solver_type, penalty=PenaltyTypes.L2, alpha=alpha)
