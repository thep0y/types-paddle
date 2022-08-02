from abc import abstractmethod
from typing import Any, Dict


class LRScheduler:
    def __init__(
        self, learning_rate: float = ..., last_epoch: int = ..., verbose: bool = ...
    ) -> None:
        ...

    def step(self, epoch: int | None = ...) -> None:
        ...

    def state_dict(self) -> Dict[str, Any]:
        ...

    def state_keys(self) -> None:
        ...

    def set_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get_lr(self) -> float:
        ...
