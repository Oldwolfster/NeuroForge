# ════════════════════════════════════════════════════════════════════════════
# ShardNeuroScope.py   COMPLETE FILE REPLACEMENT — the mission-ender.
#                      NeuroEngine, ShardLaunch, hyper: all evicted.
#                      Viewing a run no longer trains a run.
# ════════════════════════════════════════════════════════════════════════════
# ShardNeuroScope.py
from pathlib import Path

from src.NNA.TriFromDisk        import TriFromDisk
from src.NNA.utils.db_conn      import db_conn
from src.NeuroForge.NeuroForge  import NeuroForge


class ShardNeuroScope:
    """Replay-only. Resurrects runs from batch_history and shows them.
       No engine, no retrain -- the db IS the model's memory."""

    def __init__(self, db_path, run_ids_arg):
        self.db_path = Path(db_path).resolve()
        self.run_ids = [int(x) for x in run_ids_arg.split(",")]

    def run(self):
        db = db_conn(self.db_path)
        NeuroForge([TriFromDisk(db, rid) for rid in self.run_ids])

