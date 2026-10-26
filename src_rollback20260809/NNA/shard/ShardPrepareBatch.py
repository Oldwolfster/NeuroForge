# ShardPrepareBatch.py

import json
from pathlib import Path

from src.ArenaSettings import HyperParameters
from src.NNA.engine.BatchCreator import BatchCreator
from src.NNA.legos._LegoManager import LegoManager
from src.NNA.utils.db_conn import db_conn


class ShardPrepareBatch:

    def __init__(self, db_path):
        self.db_path = Path(db_path).resolve()
        self.db      = db_conn(self.db_path)
        self.conn    = self.db.conn

    def prepare(self, batch_id):
        config = self.read_config(batch_id)
        hyper  = self.build_hyper(config)
        BatchCreator(hyper).create_new_batch(batch_id=batch_id)
        print(f"prepare_batch complete: batch_id={batch_id}")

    def read_config(self, batch_id):
        raw = self.db.query_value("SELECT config FROM batch WHERE batch_id = ?", (batch_id,))
        if not raw: raise RuntimeError(f"HOUSTON: no batch row for batch_id={batch_id}")
        return json.loads(raw)

    def build_hyper(self, config):
        hyper                   = HyperParameters()
        hyper.nf_db             = self.db
        hyper.gladiators        = config["gladiators"]
        hyper.arenas            = config["arenas"]
        hyper.epochs_to_run     = config.get("epochs_to_run",     hyper.epochs_to_run)
        hyper.training_set_size = config.get("training_set_size", hyper.training_set_size)
        hyper.record_level      = config.get("record_level")

        dimensions              = {}
        dimensions["seed"]      = config["seeds"]

        lr = config.get("learning_rate", 0)
        if lr and lr != 0:
            dimensions["learning_rate"] = [lr]

        raw_dims = config.get("dimensions", {})
        if raw_dims:
            dimensions.update(self.resolve_dimensions(raw_dims))

        hyper.dimensions = dimensions
        return hyper

    def resolve_dimensions(self, raw_dims):
        mgr      = LegoManager()
        resolved = {}
        for key, values in raw_dims.items():
            if not isinstance(values, list):
                values = [values]
            if mgr.is_lego_dimension(key):
                resolved[key] = [mgr.resolve_lego_by_name(key, v) for v in values]
            else:
                resolved[key] = values
        return resolved

