
# ════════════════════════════════════════════════════════════════════════════
# TriFromDisk.py   NEW FILE  (src/NNA/engine/)
# ════════════════════════════════════════════════════════════════════════════
# TriFromDisk.py
import ast
from types import SimpleNamespace

from src.NNA.engine.BinaryDecision      import BinaryDecision
from src.NNA.engine.Config              import Config
from src.NNA.engine.ReplayTrainingData  import ReplayTrainingData
from src.NNA.utils.enums                import RecordLevel

TRAIN, VALIDATE, TEST = 0, 1, 2


class TriFromDisk:
    """
    Duck-typed TrainingRunInfo resurrected from batch_history by run_id.
    NeuroScope cannot tell it from a live TRI -- BinaryDecision and Config
    are the REAL classes, rebuilt from stored values, so display logic
    and replay logic can never drift apart.
    """
    def __init__(self, db, run_id: int):
        self.db             = db
        self.run_id         = run_id
        row                 = self.load_row()
        self.hydrate_run     (row)
        self.hydrate_phases  (row)
        self.training_data  = ReplayTrainingData(row)
        self.data           = self.build_phase_stubs(row)
        self.BD             = BinaryDecision(self.training_data)
        self.config         = self.build_config(row)

    # ---------------------------------------------------------------- loading

    def load_row(self) -> dict:
        cur = self.db.conn.execute("SELECT * FROM batch_history WHERE run_id = ?", (self.run_id,))
        rec = cur.fetchone()
        if rec is None: raise ValueError(f"run_id {self.run_id} not found in batch_history")
        return dict(zip([d[0] for d in cur.description], rec))

    def hydrate_run(self, row):
        self.gladiator          = row['gladiator']
        self.seed               = row['seed']
        self.last_epoch         = row['epoch_count']
        self.checkpoint_phase   = row['checkpoint_phase']
        self.converge_cond      = row['convergence_condition']
        self.time_seconds       = row['runtime_seconds']
        self.record_level       = RecordLevel(row['record_level'])
        self.backprop_headers   = ReplayTrainingData.json_or_none(row['backprop_headers'])
        self.scaler_params      = ReplayTrainingData.json_or_none(row['scaler_params'])

    def hydrate_phases(self, row):
        """Train/validate keep only their best MAE. Accuracy is the verdict, so it exists
           on test alone -- the other slots stay None rather than carry a plausible wrong number."""
        acc                      = row['accuracy']
        self.lowest_mae          = [row['mae_train'],       row['mae_validate'],       row['mae']]
        self.lowest_mae_epoch    = [row['mae_epoch_train'], row['mae_epoch_validate'], None]
        self.last_mae            = [None,                   None,                      row['mae']]
        self.best_accuracy       = [None,                   None,                      acc]
        self.best_accuracy_epoch = [None,                   None,                      None]
        self.last_accuracy       = [None,                   None,                      acc]

    def build_phase_stubs(self, row) -> list:
        """PopupArchitecture only asks data[phase] for None-ness and sample_count."""
        counts = [row['sample_count_train'], row['sample_count_validate'], row['sample_count_test']]
        return [SimpleNamespace(sample_count=c) if c is not None else None for c in counts]


    def phase_ran(self, phase: int) -> bool:
        return self.data[phase] is not None and self.lowest_mae[phase] is not None

    # ---------------------------------------------------------------- config

    def build_config(self, row) -> Config:
        cfg = Config(self)                  # real Config -- reads training_data.input_count
        self.restore_legos(cfg, row)
        self.restore_plain(cfg, row)
        cfg.finish_setup()                  # rebuilds the MultiScaler exactly as live did
        return cfg

    def restore_legos(self, cfg, row):
        for key in ('optimizer', 'weight_initializer', 'loss_function',
                    'hidden_activation', 'output_activation', 'target_scaler'):
            setattr(cfg, key, cfg.deserialize_lego(row[key]))
        cfg.input_scalers = self.parse_input_scalers(cfg, row['input_scalers'])

    def restore_plain(self, cfg, row):
        cfg.learning_rate   = row['learning_rate']
        cfg.batch_size      = row['batch_size']
        cfg.architecture    = ast.literal_eval(row['architecture'])

    def parse_input_scalers(self, cfg, raw):
        """Stored as json {field: name} map, a single lego name, or NULL."""
        if raw is None:                     return None
        if raw.strip().startswith('{'):     return ReplayTrainingData.json_or_none(raw)
        return cfg.deserialize_lego(raw)

    # ---------------------------------------------------------------- convenience

    @classmethod
    def latest(cls, db):
        """Most recent replayable run -- handy for smoke testing."""
        sql = """SELECT run_id FROM batch_history
                 WHERE  status = 'completed' AND record_level >= 2
                 ORDER  BY run_id DESC LIMIT 1"""
        rec = db.conn.execute(sql).fetchone()
        if rec is None: raise ValueError("No replayable runs on disk (need RecordLevel FULL+)")
        return cls(db, rec[0])


# ════════════════════════════════════════════════════════════════════════════
# Round-trip smoke test  (wherever you launch from -- 2 lines)
# ════════════════════════════════════════════════════════════════════════════
#   from src.NNA.engine.TriFromDisk import TriFromDisk
#   NeuroForge([TriFromDisk.latest(db)])        # db = your db_conn, no NNA run needed