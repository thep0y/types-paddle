from .lr import LRScheduler
from ..fluid.regularizer import WeightDecayRegularizer
from ..fluid.clip import GradientClipBase


class Optimizer:
    def __init__(
        self,
        learning_rate: float | LRScheduler,
        parameters: tuple | list | None = ...,
        weight_decay: float | WeightDecayRegularizer = ...,
        grad_clip: GradientClipBase | None = ...,
        name: str | None = ...,
    ) -> None:
        ...
