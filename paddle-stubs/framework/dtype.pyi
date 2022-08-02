from ..fluid.core import VarDesc

dtype = VarDesc.VarType
dtype.__qualname__ = "dtype"
dtype.__module__ = "paddle"

uint8: dtype
int8: dtype
int16: dtype
int32: dtype
int64: dtype

float32: dtype
float64: dtype
float16: dtype
bfloat16: dtype

complex64: dtype
complex128: dtype

bool: dtype
