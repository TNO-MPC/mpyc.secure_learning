from typing import List, Optional, Union

import numpy as np

Number = Union[int, float]
Array = Union[np.ndarray, List[List[Number]]]
Vector = Union[np.ndarray, List[Number]]

def r2_score(
    y_true: Union[Array, Vector],
    y_pred: Union[Array, Vector],
    *,
    sample_weight: Optional[Vector] = ...,
    multioutput: Optional[str] = ...
) -> float: ...
def mean_squared_error(
    y_true: Union[Array, Vector],
    y_pred: Union[Array, Vector],
    *,
    sample_weight: Optional[Vector] = ...,
    multioutput: Union[str, Vector] = ...,
    squared: bool = ...
) -> float: ...
