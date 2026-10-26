# ══════════════════════════════════════════════════════════════════════
# ShardLaunch.py  —  NEW FILE  —  the ONE way to build hyper and run a batch
#   Every launch path (parallel training shard, NeuroScope replay) goes
#   through here. Callers differ ONLY in their genuine parameters; the
#   hyper-build + config-apply + engine-run spine is identical, so a
#   rerun CANNOT diverge from the original. Idempotency by construction.
# ══════════════════════════════════════════════════════════════════════
import json
from src.ArenaSettings import HyperParameters
from src.NNA.engine.NeuroEngine import NeuroEngine

from src.NNA.utils.db_conn import db_conn


class ShardLaunch:
    """Single source of truth for building hyper from batch config and running the engine.
    Callers pass only what genuinely differs (reporter, shard, run_ids, nf_count)."""

    @classmethod
    def build_hyper(cls, db_path, batch_id, nf_count=None, run_ids=None, reporter=None):
        """The ONE hyper build. Config first; only explicit caller params override it."""
        hyper              = HyperParameters()
        hyper.nf_db        = db_conn(db_path)
        cls.apply_config(hyper, batch_id)
        if nf_count is not None: hyper.nf_count    = nf_count
        if run_ids  is not None: hyper.neuro_FORGE = run_ids
        if reporter is not None: hyper.reporter    = reporter
        return hyper

    @classmethod
    def apply_config(cls, hyper, batch_id):
        """Read batch.config and apply onto hyper — the sole reader, so no path can drift."""
        raw = hyper.nf_db.query_value("SELECT config FROM batch WHERE batch_id = ?", (batch_id,))

        if not raw: return
        config = json.loads(raw)
        skip   = {"gladiators", "arenas", "seeds", "dimensions", "input_scalers"}
        for key, value in config.items():
            if key in skip:         continue
            if hasattr(hyper, key): setattr(hyper, key, value)
        hyper.input_scalers = config.get("input_scalers", {})

    @classmethod
    def read_batch_id(cls, db, run_id):
        """batch_id from the run itself — launch_config is dead."""
        batch_id = db.query_value("SELECT batch_id FROM batch_history WHERE run_id = ?", (run_id,))
        if batch_id is None: raise RuntimeError(f"HOUSTON: run_id {run_id} not found in batch_history")
        return batch_id