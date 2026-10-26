# ShardDispatcher.py

import argparse

from src.NNA.shard.ShardNeuroScope import ShardNeuroScope
from src.NNA.shard.ShardPrepareBatch import ShardPrepareBatch
from src.NNA.shard.ShardRunBatch import ShardRunBatch
from src.NNA.utils.db_conn import db_conn


class ShardDispatcher:

    def dispatch(self):
        args = self.parse_args()

        if   args.mode == "create_db":      self.create_db(args.db)
        elif args.mode == "prepare_batch":  self.prepare_batch(args.db, args.batch_id)
        elif args.mode == "run_batch":      self.run_batch(args.db, args.shard, args.batch_id)
        elif args.mode == "neuroscope":     self.neuro_scope(args.db, args.run_ids)


    def parse_args(self):
        parser = argparse.ArgumentParser(description="NeuroForge NNA Engine")
        parser.add_argument("--mode",     choices=["run_batch", "neuroscope", "create_db", "prepare_batch"])
        parser.add_argument("--db",       help="Path to .nf file")
        parser.add_argument("--shard",    help="e.g. 0,7 means worker 0 of 7")
        parser.add_argument("--run_ids",  help="Comma-separated run IDs for neuroscope")
        parser.add_argument("--batch_id", help="Batch row minted by the launcher")
        return parser.parse_args()

    def create_db(self, db_path):

        db_conn.open_project(db_path)

    def prepare_batch(self, db_path, batch_id):

        ShardPrepareBatch(db_path).prepare(int(batch_id))

    def run_batch(self, db_path, shard_arg, batch_id):
        ShardRunBatch(db_path, shard_arg, int(batch_id)).run()

    def neuro_scope(self, db_path, run_ids_arg):

        ShardNeuroScope(db_path, run_ids_arg).run()