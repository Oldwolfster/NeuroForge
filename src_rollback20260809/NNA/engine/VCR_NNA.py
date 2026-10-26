
# ══════════════════════════════════════════════════════════════════════════════
# VCR_NNA.py        REPLACE ENTIRE FILE — says what happened, not how it lands
# ══════════════════════════════════════════════════════════════════════════════
from typing import TYPE_CHECKING

from src.NNA.engine.Neuron       import Neuron
from src.NNA.engine.RecordSample import RecordSample
from src.NNA.utils.db_recorder   import db_recorder
from src.NNA.utils.enums         import RecordLevel

if TYPE_CHECKING: from src.NNA.engine.TrainingRunInfo import TrainingRunInfo


class VCR_NNA:
    """Decides what gets recorded.  db_recorder decides how it lands."""

    KEY_FIELDS = ("run_id", "epoch", "sample_id", "nid", "weight_id")

    def __init__(self, TRI):
        self.TRI      : TrainingRunInfo = TRI
        self.recorder                   = db_recorder(TRI.db)

    # VCR_NNA.py  method: write_epoch  Update:  buffer + flush, no hand-built SQL
    def write_epoch(self, record_epoch):
        self.recorder.buffer_object("RecordEpoch", record_epoch)
        self.recorder.flush()
        if not self.TRI.should_record(RecordLevel.SUMMARY):  return
        if self.TRI.hyper.reporter: self.TRI.hyper.reporter.report_epoch(record_epoch.mae)

    # VCR_NNA.py  method: write_sample  Update:  three helpers, no SQL
    def write_sample(self, record_sample: RecordSample, blame_calculations, phase):
        self.recorder.buffer_object("RecordSample", record_sample)
        if phase != 0: return                           # validation and test filtered for now
        self.record_blame_calculations(blame_calculations)
        self.record_neurons(record_sample)
        self.record_weights(record_sample)

    # VCR_NNA.py  method: record_neurons  NEW:  one row per neuron per sample
    def record_neurons(self, record_sample):
        for layer in Neuron.layers:
            for neuron in layer:
                self.recorder.buffer_object("Neuron", neuron,
                    run_id    = self.TRI.run_id,
                    epoch     = record_sample.epoch,
                    sample_id = record_sample.sample_id)

    # VCR_NNA.py  method: record_weights  NEW:  was bulk_insert_weights + f-string SQL
    def record_weights(self, record_sample):
        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id, (before, value) in enumerate(zip(neuron.weights_before,
                                                                neuron.weights)):
                    self.recorder.buffer_row("Weight", (
                        self.TRI.run_id, record_sample.epoch, record_sample.sample_id,
                        neuron.nid, weight_id, before, value))

    # VCR_NNA.py  method: record_blame_calculations  Update:  buffer, conversion now applied
    def record_blame_calculations(self, blame_calculations):
        if not self.TRI.should_record(RecordLevel.FULL): return
        for row in blame_calculations:
            self.recorder.buffer_row("ErrorSignalCalcs", row)
        blame_calculations.clear()

    # VCR_NNA.py  method: record_optimizer_logic  Update:  buffer, headers unchanged
    def record_optimizer_logic(self, record: dict):
        if not self.TRI.should_record(RecordLevel.FULL): return
        if self.TRI.backprop_headers is None:
            self.TRI.backprop_headers = [k for k in record.keys()
                                         if k not in self.KEY_FIELDS]
        self.recorder.buffer_row("WeightAdjustments", self.pad_args(record))

    # VCR_NNA.py  method: pad_args  NEW:  keys first, then args padded to ARG_COUNT
    def pad_args(self, record):
        args = [v for k, v in record.items() if k not in self.KEY_FIELDS]
        args = (args + [None] * db_recorder.ARG_COUNT)[:db_recorder.ARG_COUNT]
        return tuple(record[k] for k in self.KEY_FIELDS) + tuple(args)