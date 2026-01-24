from enum import Enum

from src.ArenaSettings import HyperParameters
from src.NNA.engine.RecordEpoch import RecordEpoch
from src.NNA.engine.RecordSample import RecordSample
from src.NNA.engine.VCR_NNA import VCR_NNA
from src.NNA.engine.early_stopping.EarlyStopper import EarlyStopper
from src.NNA.utils import RamDB
from datetime import datetime

from src.NNA.engine.BinaryDecision import BinaryDecision
from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingData import TrainingData
from src.NNA.utils.enums import RecordLevel


class TrainingRunInfo:
    def __init__(self, hyper: HyperParameters, training_data: TrainingData, setup: dict, record_level: RecordLevel, run_id: int):
        #Context-style objects
        self.record_level:      RecordLevel         = record_level
        self.db:                RamDB               = hyper.db_ram
        self.training_data:     TrainingData        = training_data
        self.hyper:             HyperParameters     = hyper
        self.config:            Config              = Config(self)
        self.BD:                BinaryDecision      = BinaryDecision(training_data)
        self.vcr_nna:           VCR_NNA             = VCR_NNA(self)  #reference set in base gladiator.
        self.backprop_headers:  list                = None
        self.setup:             dict                = setup                 #the string written to db with purpose of rerunning exactly at a later date
        self.early_stopper:     EarlyStopper        = EarlyStopper(self)

        #Non- Training info
        self.gladiator:         str                 = setup["gladiator"]
        self.run_id:            int                 = run_id
        self.seed:              int                 = setup.get("seed")
        self.time_start:        datetime            = datetime.now()
        self.time_end:          datetime            = None
        self.explore_epochs:    int                 = -1

        # Training Metrics
        self.abs_err_for_epoch: float               = 0
        self.bd_correct:        int                 = 0

        #Summary Metrics
        self.last_epoch:        int                 = 0
        self.converge_cond:     str                 = None
        self.last_mae:          float               = 0
        self.lowest_mae:        float               = 6.9e69
        self.lowest_mae_epoch:  int                 = 0
        self.best_accuracy:     float               = -1.0
        self.best_accuracy_epoch: int               = 0

    def record_finish_time(self):
        self.time_end = datetime.now()

    def should_record(self, minimum_level: RecordLevel) -> bool:
        #return True
        return self.record_level.value >= minimum_level.value

    def get_epochs(self,exploratory_epochs: int) -> int:
        self.explore_epochs = exploratory_epochs=exploratory_epochs
        if exploratory_epochs == 0: return self.hyper.epochs_to_run
        return exploratory_epochs


    @property
    def time_seconds(self) -> float:
        if self.time_start is not None and self.time_end is not None:
            return (self.time_end - self.time_start).total_seconds()
        return -1.0

    @property
    def accuracy_regression(self) -> float:
        """Returns regression accuracy as percentage: 100 * (1 - MAE/mean_target), clamped to 0-100."""

        mean_target = self.training_data.mean_absolute_target
        #print(f"mean_target: {mean_target} self.last_mae: {self.last_mae}   ")
        if self.last_mae == 0: return 100.0
        if mean_target   == 0: return 0.0
        return  (1.0 - (self.last_mae / mean_target)) * 100

    @property
    def accuracy_bd(self) -> float:
        bd_correct = self.bd_correct
        samples = self.training_data.sample_count
        return (bd_correct / samples ) * 100

    @property
    def accuracy(self) -> float:
        if self.training_data.is_binary_decision:   return self.accuracy_bd
        else:                                       return self.accuracy_regression

#############################################################################
############################################################################
    def record_sample(self, record_sample: RecordSample, blame_calculations):
        self.abs_err_for_epoch  +=abs(record_sample.error_unscaled)
        self.bd_correct         += record_sample.is_true
        if self.should_record(RecordLevel.FULL):
            self.vcr_nna.write_sample(record_sample, blame_calculations)

    def record_epoch(self, epoch: int):
        self.last_epoch = epoch
        self.last_mae = self.abs_err_for_epoch / self.training_data.sample_count

        if self.lowest_mae > self.last_mae:
            self.lowest_mae = self.last_mae
            self.lowest_mae_epoch       = epoch

        if self.accuracy > self.best_accuracy:
            self.best_accuracy_epoch = epoch
            self.best_accuracy       = self.accuracy

        epoch_record = RecordEpoch(
            run_id=self.run_id,
            epoch=epoch,
            correct=self.bd_correct,
            wrong=self.training_data.sample_count - self.bd_correct,
            accuracy=self.accuracy,
            mae=self.last_mae,
        )
        self.vcr_nna.write_epoch(epoch_record)
        self.converge_cond = "No Early Stopping"
        if self.explore_epochs == 0:  self.converge_cond      = self.early_stopper.check_if_converged(epoch_record.epoch,epoch_record.mae) # "Did Not Converge" # TODO: convergence detector
        self.abs_err_for_epoch  = 0
        self.bd_correct         = 0
        return   self.converge_cond
