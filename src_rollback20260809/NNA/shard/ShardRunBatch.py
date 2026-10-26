# ShardRunBatch.py

import json
from pathlib import Path

from src.ArenaSettings import HyperParameters
from src.NNA.engine.NeuroEngine import NeuroEngine
from src.NNA.shard.ShardLaunch import ShardLaunch
from src.NNA.shard.ShardReporter import ShardReporter
from src.NNA.utils.RamDB import RamDB


class ShardRunBatch:

    def __init__(self, db_path, shard_arg, batch_id):
        self.db_path           = Path(db_path).resolve()
        self.shard, self.total = self.parse_shard(shard_arg)
        self.batch_id          = batch_id

    # ShardRunBatch.py  method: run  Update: uses self.batch_id
    def runOOOLLLDDD(self):
        hyper            = self.build_hyper()
        hyper.reporter   = ShardReporter(self.derive_base_path(), self.shard)
        engine = NeuroEngine(hyper)
        engine.run_a_batch(batch_id=self.batch_id, shard=(self.shard, self.total))

    def run(self):
        reporter = ShardReporter(self.derive_base_path(), self.shard)
        hyper    = ShardLaunch.build_hyper(self.db_path, self.batch_id, reporter=reporter)
        NeuroEngine(hyper).run_a_batch(batch_id=self.batch_id, shard=(self.shard, self.total))





    def build_hyper(self):
        hyper            = HyperParameters()
        hyper.db_dsk     = RamDB(self.db_path)

        hyper.nf_count   = 0
        self.apply_batch_config(hyper)
        return hyper

    def apply_batch_configDELETEME(self, hyper):
        row = hyper.db_dsk.conn.execute("SELECT config FROM batch WHERE batch_id = ?", (self.batch_id,)).fetchone()
        if not row: return
        config = json.loads(row[0])
        skip   = {"gladiators", "arenas", "seeds", "dimensions", "input_scalers"}
        for key, value in config.items():
            if key in skip:         continue
            if hasattr(hyper, key): setattr(hyper, key, value)
        hyper.input_scalers = config.get("input_scalers", {})    # whole map; Config picks its arena's slice


    def read_batch_id(self, conn):
        cursor = conn.cursor()
        cursor.execute('''
            SELECT batch_id FROM launch_config
            WHERE batch_id IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        if not row: raise RuntimeError("No launch_config with batch_id found.")
        return row[0]

    def parse_shard(self, shard_arg):
        parts = shard_arg.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid shard format '{shard_arg}'. Expected 'N,TOTAL'")
        return int(parts[0]), int(parts[1])

    def derive_base_path(self):
        return str(self.db_path.parent.parent)