import numpy as np

from typing import List, Literal, Optional, Tuple, TypeVar, Union
from numpy import (
    ndarray,
    bool_,
    ubyte,
    ushort,
    uintc,
    uint,
    ulonglong,
    byte,
    short,
    intc,
    int_,
    longlong,
    half,
    single,
    double,
    longdouble,
    csingle,
    cdouble,
    clongdouble,
    datetime64,
    timedelta64,
    object_,
    str_,
    bytes_,
    void,
)
from ..fluid.core import CPUPlace, CUDAPinnedPlace, CUDAPlace

from ..framework import Tensor

ScalarType = TypeVar(
    "ScalarType",
    int,
    float,
    complex,
    bool,
    bytes,
    str,
    memoryview,
    bool_,
    csingle,
    cdouble,
    clongdouble,
    half,
    single,
    double,
    longdouble,
    byte,
    short,
    intc,
    int_,
    longlong,
    timedelta64,
    datetime64,
    object_,
    bytes_,
    str_,
    ubyte,
    ushort,
    uintc,
    uint,
    ulonglong,
    void,
)
ShapeType = Union[Tuple[float], List[float]]
ShapesType = Union[Tuple[ShapeType, ...], List[ShapeType]]

DtypeStr = Literal[
    "bool",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "complex64",
    "complex128",
]


def to_tensor(
    data: Union[ScalarType, ShapeType, ShapesType, ndarray, Tensor],
    dtype: Optional[Union[DtypeStr, np.dtype]] = ...,
    place: Optional[Union[CPUPlace, CUDAPlace, CUDAPinnedPlace, str]] = ...,
    stop_gradient: bool = ...,
) -> Tensor:
    ...


__all__ = ["to_tensor"]
