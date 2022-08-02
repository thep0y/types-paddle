from collections.abc import Generator
from typing import Dict, List, Optional
from .core import ProgramDesc
from ..framework import Tensor


def is_compiled_with_xpu() -> bool:
    ...


def is_compiled_with_npu() -> bool:
    ...


def is_compiled_with_cinn() -> bool:
    ...


def is_compiled_with_cuda() -> bool:
    ...


def is_compiled_with_rocm() -> bool:
    ...


class Block:
    def __init__(self, program: Program, idx: int) -> None:
        ...

    def to_string(self, throw_on_error: bool, with_details: bool = ...) -> str:
        ...

    @property
    def parent_idx(self) -> int:
        ...

    @property
    def forward_block_idx(self) -> int:
        ...

    @property
    def backward_block_idx(self) -> int:
        ...

    @property
    def idx(self) -> int:
        ...


class Variable:
    def __init__(
        self,
        block: Block,
        type,
        name: Optional[str] = ...,
        dtype=None,
        lod_level=None,
        capacity=None,
        persistable=None,
        error_clip=None,
        stop_gradient=False,
        is_data=False,
        need_check_feed=False,
        belong_to_optimizer=False,
        **kwargs,
    ) -> None:
        ...


class Program:
    desc: ProgramDesc

    def global_seed(self, seed: int = ...) -> None:
        ...

    def to_string(self, throw_on_error: bool, with_details: bool = ...) -> str:
        ...

    def clone(self, for_test: bool = ...) -> Program:
        ...

    @staticmethod
    def parse_from_string(binary_str: str) -> Program:
        ...

    @property
    def random_seed(self) -> int:
        ...

    @property
    def num_blocks(self) -> int:
        ...

    @random_seed.setter
    def random_seed(self, seed: int):
        ...

    def global_block(self) -> Block:
        ...

    def block(self, index: int) -> Block:
        ...

    def current_block(self) -> Block:
        ...

    def list_vars(self) -> Generator[Tensor, None, None]:
        ...

    def all_parameters(self) -> List[Parameters]:
        ...

    def state_dict(
        self, mode: str = "all", scope: Optional[Scope] = None
    ) -> Dict[str, Tensor]:
        ...

    def set_state_dict(
        self, state_dict: Dict[str, Tensor], scope: Optional[Scope] = None
    ):
        ...
