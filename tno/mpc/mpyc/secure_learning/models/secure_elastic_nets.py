"""
Implementation of ElasticNets regression model.
"""
from tno.mpc.mpyc.secure_learning.models.secure_linear import Linear
from tno.mpc.mpyc.secure_learning.models.secure_model import PenaltyTypes, SolverTypes


class ElasticNets(Linear):
    r"""
    Solver for Elastic Nets regression. Optimizes a model with objective
    function:
    $$\frac{1}{2{n}_{\textrm{samples}}} \times ||y - X_times_w||^2_2 + \alpha_1 ||w||_1 + \frac{\alpha_2 ||w||^2_2}{2}$$
    """
    name = "Elastic nets regression"

    def __init__(
        self,
        solver_type: SolverTypes = SolverTypes.GD,
        alpha1: float = 1,
        alpha2: float = 1,
    ) -> None:
        """
        Constructor method.

        :param solver_type: Solver type to use (e.g. Gradient Descent aka GD)
        :param alpha1: Regularisation parameter for L2
        :param alpha2: Regularisation parameter for L2
        """
        super().__init__(
            solver_type, penalty=PenaltyTypes.ELASTICNET, alpha1=alpha1, alpha2=alpha2
        )
