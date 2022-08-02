from io import BytesIO
from typing import Any, Union


def load(
    path: Union[str, BytesIO],
    model_filename: str = ...,
    params_filename: str = ...,
    return_numpy: bool = ...,
) -> Any:
    ...


def save(
    obj: Any, path: str | BytesIO, protocol: int = ..., use_binary_format: bool = ...
) -> None:
    ...
