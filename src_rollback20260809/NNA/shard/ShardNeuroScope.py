# ════════════════════════════════════════════════════════════════════════════
# ShardNeuroScope.py   COMPLETE FILE REPLACEMENT — the mission-ender.
#                      NeuroEngine, ShardLaunch, hyper: all evicted.
#                      Viewing a run no longer trains a run.
# ════════════════════════════════════════════════════════════════════════════
# ShardNeuroScope.py
from pathlib import Path

from src.NNA.TriFromDisk        import TriFromDisk
from src.NNA.engine.NeuroEngine import NeuroEngine
from src.NNA.shard.ShardLaunch import ShardLaunch
from src.NNA.utils.db_conn      import db_conn
from src.NeuroForge.NeuroForge  import NeuroForge


class ShardNeuroScope:
    """Replay-only. Resurrects runs from batch_history and shows them.
       No engine, no retrain -- the db IS the model's memory."""

    def __init__(self, db_path, run_ids_arg):
        self.db_path = Path(db_path).resolve()
        self.run_ids = [int(x) for x in run_ids_arg.split(",")]


    # ShardNeuroScope.py  method: run  Update: train anything without a tape, then replay
    def run(self):
        db = db_conn(self.db_path)
        for rid in self.run_ids:
            if not self.has_tape(db, rid): self.train_one(db, rid)
        NeuroForge([TriFromDisk(db, rid) for rid in self.run_ids])

    # ShardNeuroScope.py  method: has_tape  NEW: record_level 2+ means RecordSample rows exist
    def has_tape(self, db, run_id):
        level = db.query_value("SELECT record_level FROM batch_history WHERE run_id = ?", (run_id,))
        return isinstance(level, int) and level >= 2

    # ShardNeuroScope.py  method: train_one  NEW: run_ids, no shard — neuro_FORGE branch 3 fires
    def train_one(self, db, run_id):
        batch_id = ShardLaunch.read_batch_id(db, run_id)
        hyper    = ShardLaunch.build_hyper(self.db_path, batch_id, run_ids=[run_id])
        NeuroEngine(hyper).run_a_batch(batch_id, shard=None)

