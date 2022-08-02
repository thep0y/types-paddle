from .io import *
from typing import List, Optional, Tuple, Union
from .dtype import dtype
from numpy import ndarray

_Other = Union[Tensor, float, int]


class Tensor:  # Tensor
    def __add__(self, other: _Other) -> Tensor:
        ...

    def __sub__(self, other: _Other) -> Tensor:
        ...

    def __array__(self, dtype: Optional[dtype] = ...) -> ndarray:
        ...

    def abs(self, x: Tensor, name: Optional[str] = ...) -> Tensor:
        ...

    def acos(self, x: Tensor, name: Optional[str] = ...) -> Tensor:
        ...

    def acosh(self, x: Tensor, name: Optional[str] = ...) -> Tensor:
        ...

    def add(self, x: Tensor, y: Tensor, name: Optional[str] = ...) -> Tensor:
        ...

    def add_(self, x: Tensor, y: Tensor, name: Optional[str] = ...) -> Tensor:
        ...

    def add_n(
        self,
        inputs: Union[Tensor, List[Tensor], Tuple[Tensor]],
        name: Optional[str] = ...,
    ) -> Tensor:
        ...

    def addmm(
        self,
        input: Tensor,
        x: Tensor,
        y: Tensor,
        beta: float = ...,
        alpha: float = ...,
        name: Optional[str] = ...,
    ) -> Tensor:
        ...

    def all(
        self,
        x: Tensor,
        axis: Optional[Union[int, list, tuple]] = ...,
        keepdim: bool = ...,
        name: Optional[str] = ...,
    ) -> Tensor:
        ...
