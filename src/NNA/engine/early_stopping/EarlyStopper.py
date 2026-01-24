from typing import  TYPE_CHECKING

from src.NNA.engine.early_stopping.Signal_ClassicDecay import Signal_ClassicDecay
from src.NNA.utils.general_text import beautify_text

if TYPE_CHECKING:     from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from scipy.stats import triang
from src.NNA.engine.early_stopping.Signal__BASE import Signal__BASE



class EarlyStopper:
    def __init__(self, TRI: "TrainingRunInfo"):
        self.TRI              :"TrainingRunInfo" = TRI #provides TRI.hyper.whatever_you_want_i_add
        self.error_hist       : dict[int, float] = {}

        # Ordered list of signals - first match wins
        self.signal_names: list[str] = [
            "Signal_ClassicDecay",
            "Signal_PerfectAccuracy",
            # Add more signal names here in priority order
        ]

        self.signals: list[Signal__BASE] = []
        self.initialize_signals()

    def initialize_signals(self):
        """Instantiate signals in the order specified by signal_names."""
        for signal_name in self.signal_names:
            import importlib
            module = importlib.import_module(f".{signal_name}", package="src.NNA.engine.early_stopping")
            signal_class = getattr(module, signal_name)
            self.signals.append(signal_class(self.error_hist, self.TRI))

    def record_epoch(self, epoch: int, mae: float):
        """Record epoch MAE for signal evaluation."""
        self.error_hist[epoch] = mae

    def check_if_converged(self, epoch: int, mae: float) -> str:
        """
        Check all signals in order.
        Returns: "" if no convergence, or signal_name if converged.
        """
        self.record_epoch(epoch, mae)
        for signal in self.signals:
            if signal.evaluate():
                return beautify_text(signal.signal_name.removeprefix("Signal_") )

        return "No Early Stopping"