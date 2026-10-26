from abc import ABC, abstractmethod
from typing import  TYPE_CHECKING
if TYPE_CHECKING:     from src.NNA.engine.TrainingRunInfo import TrainingRunInfo

class Signal__BASE(ABC):
    def __init__(self, error_hist: dict[int, float],TRI):
        self.threshold  = TRI.hyper.early_stopping_thresh
        self.error_hist = error_hist
        self.TRI        = TRI


    @property
    def signal_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self) -> bool:
        """Returns True if this signal detects convergence."""
        pass