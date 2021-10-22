from abc import ABCMeta
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

NumpyFloatArray = npt.NDArray[np.float_]
NumpyIntegerArray = npt.NDArray[np.int_]
NumpyObjectArray = npt.NDArray[np.object_]
NumpyNumberArray = Union[NumpyIntegerArray, NumpyFloatArray]
NumpyArrayTV = TypeVar(
    "NumpyArrayTV", NumpyObjectArray, NumpyIntegerArray, NumpyFloatArray
)

class BaseCrossValidator(metaclass=ABCMeta):
    def split(
        self,
        X: Union[List[List[Any]], npt.ArrayLike],
        y: Optional[npt.ArrayLike] = ...,
        groups: Optional[npt.ArrayLike] = ...,
    ) -> Iterable[Tuple[NumpyArrayTV, NumpyArrayTV]]: ...

class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta): ...

class KFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        shuffle: bool = ...,
        random_state: Optional[Union[int, np.random.RandomState]] = ...
    ) -> None: ...
