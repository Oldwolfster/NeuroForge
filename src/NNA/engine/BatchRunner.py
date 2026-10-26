from pathlib import Path

from src.NNA.legos._LegoManager import LegoManager


class BatchRunner:
    CONFIG_COLS = ["gladiator", "arena", "seed", "lr_specified", "learning_rate",
                   "batch_size", "architecture", "optimizer", "weight_initializer",
                   "loss_function", "hidden_activation", "output_activation",
                   "target_scaler", "input_scalers", "record_level"]

    def __iter__(self):
        """Make BatchRunner iterable"""
        return self

# BatchRunner.py — replace __init__ signature and the load logic

    def __init__(self, batch_id: int, hyper, shard=None):
        self.lego_mgr       = LegoManager()
        self.batch_id       = batch_id
        self.conn           = hyper.nf_db.conn
        self.current_index  = -1
        self.run_id = None
        if shard is not None:               self.load_pending_run_ids_for_shard(shard[0], shard[1])
        elif hyper.neuro_FORGE[0] == 0:     self.load_pending_run_ids()
        else:                               self.pending_run_ids = hyper.neuro_FORGE

    def load_pending_run_ids(self):
        """Load only the run_ids, not the full configs"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT run_id 
            FROM batch_history
            WHERE batch_id = ? AND status = 'pending'
            ORDER BY run_id
        ''', (self.batch_id,))

        self.pending_run_ids = [row[0] for row in cursor.fetchall()]
        print(f"🔍 BatchRunner: Found {len(self.pending_run_ids)} pending runs for batch_id={self.batch_id}")

    def __next__(self):
        """Fetch next run on-demand and return (run_id, setup)"""
        self.current_index += 1
        if self.current_index >= len(self.pending_run_ids): raise StopIteration
        self.run_id = self.pending_run_ids[self.current_index]
        print(f"@@RUN_ID={self.run_id}")
        return (self.run_id, self.load_run_config(self.run_id))

    def load_run_config(self, run_id):
        cols = ", ".join(self.CONFIG_COLS)
        row  = self.conn.execute(f"SELECT {cols} FROM batch_history WHERE run_id = ?", (run_id,)).fetchone()
        return self.deserialize_run_config(dict(zip(self.CONFIG_COLS, row)))

    def deserialize_run_config(self, run_config: dict) -> dict:
        """Convert column values back to proper types"""
        setup = {}
        for key, value in run_config.items():
            if   value is None:                          setup[key] = None
            elif key == 'architecture':                  setup[key] = eval(value)   # "[4,2,1]" -> [4,2,1]
            elif key == 'lr_specified':                  setup[key] = bool(value)
            elif key in ('seed', 'batch_size', 'record_level'): setup[key] = int(value)
            elif key == 'learning_rate':                 setup[key] = float(value)
            elif key in ('gladiator', 'arena', 'target_scaler'): setup[key] = value
            elif self.lego_mgr.is_lego_dimension(key):   setup[key] = self.lego_mgr.string_to_lego(value)
            else:                                        setup[key] = None
        return setup

    @property
    def run_position(self):
        """1-based current run number"""
        return self.current_index + 1

    @property
    def total_runs(self):
        """Total number of runs in this batch"""
        return len(self.pending_run_ids)



    def load_pending_run_ids_for_shard(self, shard, total):
        """Load only run_ids assigned to this shard. Shard is 0-based."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT run_id 
            FROM batch_history
            WHERE batch_id = ? AND status = 'pending'
            AND run_id % ? = ?
            ORDER BY run_id
        ''', (self.batch_id, total, shard))
        self.pending_run_ids = [row[0] for row in cursor.fetchall()]
        print(
            f"🔍 BatchRunner: Shard {shard}/{total} has {len(self.pending_run_ids)} pending runs for batch_id={self.batch_id}")

    @classmethod
    def first_run_ids(cls, conn, batch_id, n):
        """One definition, so tape and display cannot drift apart."""
        if n < 1: return []
        rows = conn.execute("SELECT run_id FROM batch_history WHERE batch_id = ?"
                            " ORDER BY run_id LIMIT ?", (batch_id, n)).fetchall()
        return [row[0] for row in rows]

    # BatchRunner.py  method: display_run_ids  NEW: batch-global, so 8 shards agree without talking
    def display_run_ids(self, nf_count):
        """Which runs earn FULL tape. Asked of the batch, never of the shard."""
        return BatchRunner.first_run_ids(self.conn, self.batch_id, nf_count)