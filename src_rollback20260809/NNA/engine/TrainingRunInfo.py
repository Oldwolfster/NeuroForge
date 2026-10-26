# TrainingRunInfo.py
from datetime import datetime

from src.ArenaSettings import HyperParameters
from src.NNA.engine.BinaryDecision import BinaryDecision
from src.NNA.engine.Config import Config
from src.NNA.engine.RecordEpoch import RecordEpoch
from src.NNA.engine.RecordSample import RecordSample
from src.NNA.engine.TrainingData import TrainingData
from src.NNA.engine.VCR_NNA import VCR_NNA
from src.NNA.engine.early_stopping.EarlyStopper import EarlyStopper
from src.NNA.utils.db_conn import db_conn
from src.NNA.utils.enums import RecordLevel
from src.NNA.utils.snapshot_weights import harvest_weights

TRAIN, VALIDATE, TEST = 0, 1, 2                 # plain ints -- serialize as ints, print as ints


class TrainingRunInfo:
    """
    One TRI per run. Everything run-scoped lives here exactly once (config, hyper, db,
    BD, scaler params, early stopper, seed, run_id). Everything phase-scoped lives in a
    list indexed by phase: data, accumulators, and per-phase results.
    """
    def __init__(self, hyper: HyperParameters, training_data: TrainingData, validate_data: TrainingData,
                 testing_data: TrainingData, setup: dict, record_level: RecordLevel, batch):

        # Phase-indexed data -- set FIRST, Config reads it during construction below
        self.data:                  list                = [training_data, validate_data, testing_data]
        self.data_raw:              list                = [self.rows_of(d) for d in self.data]
        self.data_scaled:           list                = [None for _ in self.data]

        # Context-style objects
        self.record_level:          RecordLevel         = record_level
        self.db:                    db_conn             = hyper.nf_db
        self.hyper:                 HyperParameters     = hyper
        self.config:                Config              = Config(self)
        self.BD:                    BinaryDecision      = BinaryDecision(training_data)
        self.vcr_nna:               VCR_NNA             = VCR_NNA(self)
        self.backprop_headers:      list                = None
        self.setup:                 dict                = setup
        self.early_stopper:         EarlyStopper        = EarlyStopper(self)
        self.scaler_params:         list                = [{} for _ in range(training_data.input_count + 1)]

        # Non-training info
        self.gladiator:             str                 = setup["gladiator"]
        self.batch_id:              int                 = batch.batch_id
        self.run_id:                int                 = batch.run_id
        self.run_position:          int                 = batch.run_position
        self.seed:                  int                 = setup.get("seed")
        self.time_start:            datetime            = datetime.now()
        self.time_end:              datetime            = None
        self.explore_epochs:        int                 = -1

        # Per-phase accumulators -- zeroed at the close of each phase's epoch
        self.abs_err:               list                = [0.0    for _ in self.data]
        self.bd_correct:            list                = [0      for _ in self.data]

        self.last_mae:              list                = [None   for _ in self.data]
        self.last_accuracy:         list                = [None   for _ in self.data]
        self.lowest_mae:            list                = [float("inf")  for _ in self.data] #[None   for _ in self.data]
        self.lowest_mae_epoch:      list                = [None   for _ in self.data]
        self.best_accuracy:         list                = [float("-inf")  for _ in self.data]#= [None   for _ in self.data]
        self.best_accuracy_epoch:   list                = [None   for _ in self.data]


        # Run-level summary
        self.last_epoch:            int                 = 0
        self.converge_cond:         str                 = None
        self.checkpoint_phase:      int                 = VALIDATE #VALIDATE #TRAIN     # flip to VALIDATE to checkpoint on validation
        self.optimum_weights:       dict                = None
        self.run_mode:              int                 = 0         # legacy -- slated for deletion

    def rows_of(self, data):
        """One get_list() per phase at construction -- not once per epoch."""
        return None if data is None else data.get_list()

    # ---------------------------------------------------------------- data access

    @property
    def training_data(self) -> TrainingData:
        """Phase 0's data. Keeps every existing TRI.training_data caller working."""
        return self.data[TRAIN]

    def data_for_phase(self, phase: int, scaled: bool):
        """phase: 0=train 1=validate 2=test"""
        if phase < 0 or phase >= len(self.data):    raise ValueError(f"Unknown phase {phase}")
        if scaled:                                  return self.data_scaled[phase]
        return self.data_raw[phase]

    def phases_due(self, epoch: int) -> list:
        """Train always. Validate when it exists. Test never runs during training."""
        due = [TRAIN]
        if self.data[VALIDATE] is not None:  due.append(VALIDATE)
        return due

    # ---------------------------------------------------------------- scaling

    def scale_inputs(self, scaler):
        """
        Fit ONCE on training data, then transform every phase with those same params.
        Only phase 0 is ever fit -- that is what makes leakage structurally impossible
        rather than a rule someone has to remember.
        """
        scaler.fit_all(self.scaler_params, self.data_raw[TRAIN])
        for phase in range(len(self.data)):
            self.apply_scaling(scaler, phase)

    def apply_scaling(self, scaler, phase: int):
        """Transform one phase using the frozen TRAINING params. transform never writes params."""
        if self.data[phase] is None: return
        self.data_scaled[phase] = scaler.transform_all(self.scaler_params, self.data_raw[phase])

    # ---------------------------------------------------------------- run bookkeeping

    def record_finish_time(self):
        self.time_end = datetime.now()

    def should_record(self, minimum_level: RecordLevel) -> bool:
        return self.record_level.value >= minimum_level.value

    def get_epochs(self, exploratory_epochs: int) -> int:
        self.explore_epochs = exploratory_epochs
        if exploratory_epochs == 0: return self.hyper.epochs_to_run
        return exploratory_epochs

    @property
    def time_seconds(self) -> float:
        if self.time_start is not None and self.time_end is not None:
            return (self.time_end - self.time_start).total_seconds()
        return -1.0

    @property
    def is_lr_sweep(self) -> bool:
        """explore_epochs is 0 only for the real run; anything else is a sweep trial."""
        return self.explore_epochs != 0

    @property
    def optimum_epoch(self) -> int:
        """Epoch the optimum_weights came from. Derived -- never stored twice."""
        return self.lowest_mae_epoch[self.checkpoint_phase]


    def phase_ran(self, phase: int) -> bool:
        """True once a phase has closed at least one epoch."""
        return self.data[phase] is not None and self.lowest_mae_epoch[phase] is not None
    # ---------------------------------------------------------------- accuracy

    def accuracy_of(self, phase: int) -> float:
        if self.data[TRAIN].is_binary_decision:     return self.accuracy_bd_of(phase)
        return self.accuracy_regression_of(phase)

    def accuracy_bd_of(self, phase: int) -> float:
        return (self.bd_correct[phase] / self.data[phase].sample_count) * 100

    def accuracy_regression_of(self, phase: int) -> float:
        """100 * (1 - MAE/mean_target). Reads last_mae, so call only after close_phase."""
        mean_target = self.data[phase].mean_absolute_target
        if self.last_mae[phase] == 0:   return 100.0
        if mean_target          == 0:   return 0.0
        return (1.0 - (self.last_mae[phase] / mean_target)) * 100

    @property
    def accuracy(self) -> float:
        """Phase 0. Keeps the epoch print and NNA_History working unchanged."""
        return self.accuracy_of(TRAIN)

    # ---------------------------------------------------------------- recording

    def record_sample(self, record_sample: RecordSample, blame_calculations, phase: int):
        self.abs_err[phase]     += abs(record_sample.error_unscaled)
        self.bd_correct[phase]  += record_sample.is_true
        if self.should_record(RecordLevel.FULL):
            self.vcr_nna.write_sample(record_sample, blame_calculations, phase)

    def record_epoch(self, epoch: int, phase: int) -> str:
        self.close_phase        (epoch, phase)
        self.write_epoch_row    (epoch, phase)
        self.reset_accumulators (phase)
        return self.check_convergence(epoch, phase)

    def close_phase(self, epoch: int, phase: int):
        """Fold this phase's accumulators into its own results. Only phase 0 defines last_epoch."""
        if phase == TRAIN:  self.last_epoch = epoch
        self.last_mae[phase] = self.abs_err[phase] / self.data[phase].sample_count
        self.last_accuracy[phase] = self.accuracy_of(
            phase)  # after last_mae, before reset -- both readers need it here

        if self.lowest_mae[phase] > self.last_mae[phase]:
            self.lowest_mae[phase] = self.last_mae[phase]
            self.lowest_mae_epoch[phase] = epoch
            if phase == self.checkpoint_phase:  self.optimum_weights = harvest_weights()

        if self.last_accuracy[phase] > self.best_accuracy[phase]:
            self.best_accuracy[phase] = self.last_accuracy[phase]
            self.best_accuracy_epoch[phase] = epoch

    def write_epoch_row(self, epoch: int, phase: int):
        if self.is_lr_sweep: return
        self.vcr_nna.write_epoch(RecordEpoch(
            run_id      = self.run_id,
            epoch       = epoch,
            phase       = phase,
            correct     = self.bd_correct[phase],
            wrong       = self.data[phase].sample_count - self.bd_correct[phase],
            accuracy    = self.accuracy_of(phase),
            mae         = self.last_mae[phase],
        ))

    def reset_accumulators(self, phase: int):
        self.abs_err[phase]     = 0.0
        self.bd_correct[phase]  = 0

    def check_convergence(self, epoch: int, phase: int) -> str:
        """Only the checkpoint phase may stop the run -- otherwise patience counts double."""
        if phase != self.checkpoint_phase:  return self.converge_cond
        self.converge_cond = "No Early Stopping"
        if not self.is_lr_sweep:  self.converge_cond = self.early_stopper.check_if_converged(epoch, self.last_mae[phase])
        return self.converge_cond