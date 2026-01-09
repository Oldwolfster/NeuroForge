from enum import Enum

from src.ArenaSettings import HyperParameters
from src.NNA.utils import RamDB
from datetime import datetime

from src.NNA.engine.BinaryDecision import BinaryDecision
from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingData import TrainingData


class RecordLevel(Enum):
    NONE    = 0     # No recording — e.g., hyperparameter sweep (LR probe)
    SUMMARY = 1     # Basic stats: accuracy, loss, convergence, etc.
    FULL    = 2     # + Iteration history, weight deltas, etc. (NeuroForge playback)
    DEBUG   = 3     # + Diagnostics, blame signals, and dev-level traces

class TrainingRunInfo:
    def __init__(self, hyper: HyperParameters, training_data: TrainingData, setup: dict, record_level: RecordLevel, run_id: int):
        self.record_level:      RecordLevel         = record_level
        self.db: RamDB = hyper.db_ram
        self.training_data:     TrainingData        = training_data
        self.hyper:             HyperParameters     = hyper
        self.config:            Config              = Config(self)
        self.BD:                BinaryDecision      = BinaryDecision(training_data)
        self.setup:             dict                = setup                 #the string written to db with purpose of rerunning exactly at a later date
        self.gladiator:         str                 = setup["gladiator"]
        self.run_id:            int                 = run_id
        self.time_start:        datetime            = datetime.now()
        self.time_end:          datetime            = None

        self.converge_cond:     str                 = None
        self.bd_correct:        int                 = 0
        self.mae:               float               = None
        self.lowest_mae:        float               = 6.9e69
        self.lowest_mae_epoch:  int                 = 0
        self.best_accuracy:     float               = -1.0
        self.best_accuracy_epoch: int               = 0

    def record_finish_time(self):
        self.time_end = datetime.now()

    def should_record(self, minimum_level: RecordLevel) -> bool:
        #return True
        return self.record_level.value >= minimum_level.value

    @property
    def accuracy_regression(self) -> float:
        """Returns regression accuracy as percentage: 100 * (1 - MAE/mean_target), clamped to 0-100."""
        mean_target = self.training_data.mean_absolute_target
        if self.mae is None or mean_target == 0:
            return 0.0
        return max(0.0, (1.0 - (self.mae / mean_target)) * 100)

    @property
    def time_seconds(self) -> float:
        if self.time_start is not None and self.time_end is not None:
            return (self.time_end - self.time_start).total_seconds()
        return -1.0

    @property
    def accuracy_bd(self) -> float:
        bd_correct = self.bd_correct
        samples = self.training_data.sample_count
        return (bd_correct / samples ) * 100

    @property
    def accuracy(self) -> float:
        if self.training_data.is_binary_decision:
            return self.accuracy_bd
        else:
            return self.accuracy_regression

