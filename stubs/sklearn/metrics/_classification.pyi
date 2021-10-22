from typing import List, Optional, Union, overload

import numpy as np
import numpy.typing as npt

Array = Union[npt.NDArray[np.float_], npt.NDArray[np.int_], List[List[float]]]
Vector = Union[npt.NDArray[np.float_], npt.NDArray[np.int_], List[float]]

def accuracy_score(
    y_true: Vector,
    y_pred: Vector,
    *,
    normalize: bool = ...,
    sample_weight: Optional[Vector] = ...,
) -> float: ...
def precision_score(
    y_true: Vector,
    y_pred: Vector,
    *,
    labels: npt.NDArray[np.str_] = ...,
    pos_label: Union[int, str] = ...,
    average: str = ...,
    sample_weight: Optional[Vector] = ...,
    zero_division: Union[int, str] = ...,
) -> float: ...
def recall_score(
    y_true: Vector,
    y_pred: Vector,
    *,
    labels: npt.NDArray[np.str_] = ...,
    pos_label: Union[int, str] = ...,
    average: str = ...,
    sample_weight: Optional[Vector] = ...,
    zero_division: Union[int, str] = ...,
) -> float: ...
def f1_score(
    y_true: Vector,
    y_pred: Vector,
    *,
    labels: npt.NDArray[np.str_] = ...,
    pos_label: Union[int, str] = ...,
    average: str = ...,
    sample_weight: Optional[Vector] = ...,
    zero_division: Union[int, str] = ...,
) -> float: ...
