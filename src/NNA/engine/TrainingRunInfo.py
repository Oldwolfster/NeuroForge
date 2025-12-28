from enum import Enum

from src.ArenaSettings import HyperParameters
from src.NNA.engine import RamDB
from datetime import datetime

from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingData import TrainingData


class RecordLevel(Enum):
    NONE    = 0     # No recording — e.g., hyperparameter sweep (LR probe)
    SUMMARY = 1     # Basic stats: accuracy, loss, convergence, etc.
    FULL    = 2     # + Iteration history, weight deltas, etc. (NeuroForge playback)
    DEBUG   = 3     # + Diagnostics, blame signals, and dev-level traces

class TrainingRunInfo:
    def __init__(self, hyper: HyperParameters, training_data: TrainingData, setup: dict, record_level: RecordLevel):
        self.record_level:      RecordLevel         = record_level
        self.db:                RamDB               = hyper.db_ram
        self.training_data:     TrainingData        = training_data
        self.config:            Config              = Config(self)
        self.setup:             dict                = setup                 #the string written to db with purpose of rerunning exactly at a later date
        self.time_start:        datetime            = datetime.now()
        self.time_end:          datetime            = None


    def record_finish_time(self):
        self.time_end = datetime.now()

    def should_record(self, minimum_level: RecordLevel) -> bool:
        #return True
        return self.record_level.value >= minimum_level.value