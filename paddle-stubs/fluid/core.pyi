from abc import ABC, abstractmethod
from typing import Dict, List


ShapeType = List[int]


class AttrType:
    BLOCK: AttrType
    BLOCKS: AttrType
    BOOL: AttrType
    BOOLS: AttrType
    FLOAT: AttrType
    FLOATS: AttrType
    INT: AttrType
    INTS: AttrType
    LONG: AttrType
    LONGS: AttrType
    STRING: AttrType
    STRINGS: AttrType

    def __init__(self, id: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...


class Attribute(ABC):
    @abstractmethod
    def type(self) -> AttrType:
        pass

    @abstractmethod
    def index(self) -> int:
        pass


class _BaseDesc:
    def attr(self, name: str) -> Attribute:  # TODO: 不能确定
        ...

    def attr_names(self) -> List[str]:
        ...

    def has_attr(self, name: str) -> bool:
        ...

    def id(self) -> int:
        ...

    def original_id(self) -> int:
        ...

    def set_original_id(self, original_id: int) -> None:
        ...

    def remove_attr(self, name: str) -> None:
        ...

    def serialize_to_string(self) -> bytes:
        ...


class OpDesc(_BaseDesc):
    def copy_from(self, target: OpDesc) -> None:
        ...

    def type(self) -> str:
        ...

    def set_type(self, type: str) -> None:
        ...

    def input(self, name: str) -> List[str]:
        ...

    def input_names(self) -> List[str]:
        ...

    def output(self, name: str) -> List[str]:
        ...

    def output_names(self) -> List[str]:
        ...

    def set_input(self, name: str, args: List[str]) -> None:
        ...

    def set_output(self, name: str, args: List[str]) -> None:
        ...

    def remove_output(self, name: str) -> None:
        ...

    def remove_input(self, name: str) -> None:
        ...

    def input_arg_names(self) -> List[str]:
        ...

    def output_arg_names(self) -> List[str]:
        ...

    def attr_type(self, name: str) -> AttrType:
        ...

    def set_block_attr(self, name: str, block: BlockDesc) -> None:
        ...

    def set_blocks_attr(self, name: str, blocks: List[BlockDesc]) -> None:
        ...

    def set_serialized_attr(self, name: str, serialized_attr: bytes) -> None:
        ...

    def check_attrs(self) -> None:
        ...

    def infer_shape(self, block: BlockDesc) -> None:
        ...

    def infer_var_type(self, block: BlockDesc) -> None:
        ...

    def set_is_target(self, is_target: bool) -> None:
        ...

    def block(self) -> BlockDesc:
        ...

    def inputs(self) -> Dict[str, List[str]]:
        ...

    def outputs(self) -> Dict[str, List[str]]:
        ...


class VarDesc(_BaseDesc):
    class VarType:
        BF16: VarDesc.VarType
        BOOL: VarDesc.VarType
        COMPLEX128: VarDesc.VarType
        COMPLEX64: VarDesc.VarType
        FEED_MINIBATCH: VarDesc.VarType
        FETCH_LIST: VarDesc.VarType
        FP16: VarDesc.VarType
        FP32: VarDesc.VarType
        FP64: VarDesc.VarType
        INT16: VarDesc.VarType
        INT32: VarDesc.VarType
        INT64: VarDesc.VarType
        INT8: VarDesc.VarType
        LOD_RANK_TABLE: VarDesc.VarType
        LOD_TENSOR: VarDesc.VarType
        LOD_TENSOR_ARRAY: VarDesc.VarType
        PLACE_LIST: VarDesc.VarType
        RAW: VarDesc.VarType
        READER: VarDesc.VarType
        SELECTED_ROWS: VarDesc.VarType
        STEP_SCOPES: VarDesc.VarType
        STRING: VarDesc.VarType
        UINT8: VarDesc.VarType
        VOCAB: VarDesc.VarType

        def name(self) -> str:
            pass

    def __init__(self, name: str) -> None:
        ...

    def clear_is_parameter(self) -> None:
        ...

    def clear_stop_gradient(self) -> None:
        ...

    def dtype(self) -> VarType:
        ...

    def dtypes(self) -> List[VarType]:
        ...

    def element_size(self) -> int:
        ...

    def get_shape(self) -> ShapeType:
        ...

    def has_is_parameter(self) -> bool:
        ...

    def has_stop_gradient(self) -> bool:
        ...

    def name(self) -> str:
        ...

    def need_check_feed(self) -> bool:
        ...

    def persistable(self) -> bool:
        ...

    def set_dtype(self, dtype: VarType) -> None:
        ...

    def set_dtypes(self, dtypes: List[VarType]) -> None:
        ...

    def set_is_parameter(self, is_parameter: bool) -> None:
        ...

    def set_need_check_feed(self, need_check_feed: bool) -> None:
        ...

    def set_persistable(self, persistable: bool) -> None:
        ...

    def set_stop_gradient(self, stop_gradient: bool) -> None:
        ...

    def set_lod_level(self, level: int) -> None:
        ...

    def set_lod_levels(self, levels: List[int]) -> None:
        ...

    def set_name(self, name: str) -> None:
        ...

    def set_shape(self, shape: ShapeType) -> None:
        ...

    def set_shapes(self, shapes: List[ShapeType]) -> None:
        ...

    def set_type(self, type: VarType) -> None:
        ...

    def shape(self) -> ShapeType:
        ...

    def shapes(self) -> List[ShapeType]:
        ...

    def stop_gradient(self) -> bool:
        ...

    def type(self) -> VarType:
        ...


class BlockDesc:
    id: int
    parent: int

    def get_forward_block_idx(self) -> int:
        ...

    def append_op(self) -> OpDesc:
        ...

    def var(self, name: str) -> VarDesc:
        ...

    def has_var(self, name: str) -> bool:
        ...

    def has_var_recursive(self, name: str) -> bool:
        ...

    def find_var(self, name: str) -> VarDesc:
        ...

    def find_var_recursive(self, name: str) -> VarDesc:
        ...

    def all_vars(self) -> List[VarDesc]:
        ...

    def op_size(self, idx: int) -> int:
        ...

    def serialize_to_string(self) -> bytes:
        ...


class ProgramDesc:
    def append_block(self):
        ...

    def flush(self):
        ...

    def get_feed_target_names(self):
        ...

    def get_fetch_target_names(self):
        ...

    def serialize_to_string(self) -> bytes:
        ...

    def parse_from_string(self):
        ...

    def get_op_deps(self):
        ...

    def num_blocks(self) -> int:
        ...

    def block(self, idx: int) -> BlockDesc:
        ...


class CUDAPlace:
    def __init__(self, id: int) -> None:
        ...

    def get_device_id(self) -> int:
        ...


class CUDAPinnedPlace:
    pass


class CPUPlace:
    pass
