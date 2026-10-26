# ShardReporter.py

from pathlib import Path

from src.NNA.utils.general_text import smart_format


class ShardReporter:

    def __init__(self, base_path, shard):
        self.temp_dir = Path(base_path) / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.shard    = shard
        self.file     = None

    def start_run(self, run_id):
        self.close_run()
        self.file = open(self.temp_dir / f"mae_{run_id}.txt", "w")

    def report_lr_trial(self, lr, mae):
        self.file.write(f"LR {lr:.1e}, {smart_format(mae)}\n")
        self.file.flush()

    def report_best_lr(self, lr):
        self.file.write(f"BEST_LR {lr:.1e}\n")
        self.file.flush()

    def report_epoch(self, mae):
        #self.file.write(f"{mae:.6f}\n")
        self.file.write(f"{smart_format(mae)}\n")
        self.file.flush()

    def close_run(self):
        if self.file:
            self.file.write("DONE\n")
            self.file.flush()
            self.file.close()
            self.file = None

    def report_failed(self, reason="FAILED"):
        if self.file:
            self.file.write(f"FAILED {reason}\n")
            self.file.flush()
            self.file.close()
            self.file = None