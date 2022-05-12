"""
Initialization of the secure-learn package
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport
from .models import SVM as SVM
from .models import ClassWeightsTypes as ClassWeightsTypes
from .models import ElasticNets as ElasticNets
from .models import ExponentiationTypes as ExponentiationTypes
from .models import Lasso as Lasso
from .models import Linear as Linear
from .models import Logistic as Logistic
from .models import PenaltyTypes as PenaltyTypes
from .models import Ridge as Ridge
from .models import SolverTypes as SolverTypes

__version__ = "1.1.1"
