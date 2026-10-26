import math
import random
from src.ArenaSettings import HyperParameters
from src.NNA.engine.BatchCreator import BatchCreator
from src.NNA.engine.BatchRunner import BatchRunner
from src.NNA.engine.NNA_History import NNA_history
from src.NNA.engine.TrainingData import TrainingData
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo, RecordLevel
from src.NNA.engine.Utils import set_seed
from src.NNA.legos._LegoManager import LegoManager
from src.NeuroForge.NeuroForge import NeuroForge

from src.NNA.utils.dynamic_instantiate import instantiate_arena, dynamic_instantiate
#import psutil
import gc

import traceback
HOLDOUT_FLOOR = 30      # below this a split is noise; identical phases are more honest than a lottery
HOLDOUT_CAP   = 2000  # past this, extra holdout is wasted training data

class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics
    def __init__(self, hyper):
        self.hyper                          = hyper
        self.training_data                  = None
        self.lego_mgr                       = LegoManager()
        self.TRIs: list[TrainingRunInfo]    = []
        if hyper.nf_count>2: hyper.nf_count = 2
        #REMOVED TO MERGE WITH UI self.run_a_batch()
        #print(f"hyper.nf_count={hyper.nf_count}")
        #if self.TRIs:                       NeuroForge(self.TRIs)  #if self.TRIs:                       self.TRIs[0].db.copy_tables_to_permanent()


    def run_a_batch(self, batch_id=None, shard=None):
        if batch_id is None: batch_id   = BatchCreator(self.hyper).create_new_batch()
        batch                           = BatchRunner(batch_id, self.hyper, shard=shard)

        for run_id, setup in batch:
            print(f"NF - run id: {run_id} and batch_id: {batch_id}")
            print(f"\n💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪")
            print(f"💪 {batch.run_position} of {batch.total_runs}-{setup['arena']} - these settings: {setup}")
            self.check_for_clear(batch)
            if self.hyper.reporter: self.hyper.reporter.start_run(run_id)
            try:                    self.run_one(batch, batch_id, run_id, setup)
            except Exception as ex: self.run_failed(batch_id, run_id, ex)
            if self.hyper.reporter: self.hyper.reporter.close_run()
        print(f"hyper.nf_count={self.hyper.nf_count}")

    # NeuroEngine.py  method: run_one  Update: six lines, and every remaining 'batch' use is now a named API problem
    def run_one(self, batch, batch_id, run_id, setup):
        if not setup.get("lr_specified", False): setup["learning_rate"] = self.learning_rate_sweep(setup, batch, run_id)
        self.clear_run_id   (  run_id)
        record_level        =  self.resolve_record_level(batch, setup)
        TRI                 =  self.atomic_train_a_model(setup, record_level, self.hyper, batch, epochs=0)
        if record_level     == RecordLevel.FULL: self.TRIs.append(TRI)
        NNA_history.record  (  TRI, run_id, self.lego_mgr)

    # NeuroEngine.py  method: run_failed   NEW:  mark it, say it, keep going.  The shard outlives the run.
    def run_failed(self, batch_id, run_id, ex):
        """status and record_level both read 'Failed' -- two columns, two literals, nothing derived."""
        detail = f"{type(ex).__name__}: {ex}"
        print(f"\n💀💀💀 RUN {run_id} FAILED — {detail}\n{traceback.format_exc()}💀💀💀\n")
        self.hyper.nf_db.conn.execute(
            "UPDATE batch_history SET status='Failed', record_level='Failed', error_text=?,"
            " completed_at=CURRENT_TIMESTAMP WHERE run_id=? AND batch_id=?", (detail, run_id, batch_id))
        self.hyper.nf_db.conn.commit()
        if self.hyper.reporter: self.hyper.reporter.report_failed(detail)


    def atomic_train_a_model(self,setup:dict,record_level, hyper:HyperParameters, batch: BatchRunner, epochs):  # ATAM is short for  -->atomic_train_a_model

        set_seed                ( setup.get("seed"))
        train, validate, test   = self.get_data(setup["arena"], hyper.training_set_size)  # ← REPLACES 3 instantiate_arena lines
        set_seed                ( setup.get("seed"))  # Reset seed as it likely was used in training_data
        TRI                     = TrainingRunInfo(hyper, train, validate, test, setup, record_level, batch)
        NN                      = dynamic_instantiate(setup["gladiator"], 1, "coliseum/gladiators", TRI)
        NN.train                ( epochs)  # Actually train model
        return                  TRI


    def resolve_record_level(self, batch, setup):
        """What was asked for, floored by what NeuroForge needs to have a tape to play."""
        display_need = RecordLevel.FULL if batch.run_id in self.full_tape_runs(batch) else RecordLevel.SUMMARY
        asked        = setup.get("record_level")
        if asked is None:                   return display_need
        return RecordLevel(max(asked, display_need.value))

    def full_tape_runs(self, batch):
        """Sharded stays batch-global so 8 shards agree without talking. Targeted has nobody to agree with."""
        if not hasattr(self, "tape_runs"):
            targeted       = self.hyper.neuro_FORGE[0] != 0
            self.tape_runs = batch.pending_run_ids[:self.hyper.nf_count] if targeted \
                             else batch.display_run_ids(self.hyper.nf_count)
        return self.tape_runs



    def get_data(self, arena_name: str, sample_count: int):
        """
        One draw, sliced three ways. An arena that honors the ask is generative and gets
        independent thirds; one that ignores it has a finite corpus and gets split.
        Nothing declares which it is -- the returned row count answers that.
        """
        corpus = instantiate_arena(arena_name, sample_count * 3)
        corpus_len = len(corpus.get_list())

        if corpus_len == sample_count * 3:
            datasets = self.split_at(corpus, sample_count)
        else:
            holdout = min(corpus_len // 5, HOLDOUT_CAP)
            if holdout < HOLDOUT_FLOOR:
                datasets = (corpus, corpus, corpus)
            else:
                datasets = self.split_at(corpus, holdout)

        train_len = len(datasets[0].get_list())
        val_len = len(datasets[1].get_list())
        test_len = len(datasets[2].get_list())

        #print(f"Dataset Counts -> Train: {train_len} | Validate: {val_len} | Test: {test_len}")

        return datasets



    def split_at(self, corpus, holdout: int):
        """Last two blocks of `holdout` rows become validate and test; everything before is train."""
        rows           = corpus.get_list()
        validate_start = len(rows) - holdout * 2
        test_start     = len(rows) - holdout
        return (self.slice_data(corpus, rows[              : validate_start]),
                self.slice_data(corpus, rows[validate_start: test_start]),
                self.slice_data(corpus, rows[test_start    :              ]))

    def slice_data(self, corpus, rows):
        """raw_data is immutable by design, so a subset means constructing -- never mutating."""
        sliced             = TrainingData(rows, corpus.feature_labels, corpus.target_labels)
        sliced.arena_name  = corpus.arena_name
        sliced.source_code = corpus.source_code
        return sliced

    # NeuroEngine.py  --  REPLACES learning_rate_sweep
    def learning_rate_sweep(self, setup: dict, batch, run_id) -> float:
        """
        Sweep learning rates from 1e1 down to 1e-15 (logarithmically).
        Stops after `patience` consecutive rising trials. Modifies only 'learning_rate' in setup.
        """
        conn = self.hyper.nf_db.conn
        #batch_id = batch.batch_id
        start_lr = 1e1
        min_lr = 1e-15
        factor = 0.1
        patience = 3  # consecutive rising trials before quitting (walked out of the crater and started uphill)
        error_rising = 0
        # ───────────────────────────────────────────────────────────────
        self.lr_log_clear(conn,  run_id)
        best_error = float("inf")
        last_error = None  # no predecessor yet -- trial 1 cannot be a rise
        best_lr = None
        lr = start_lr
        seq = 0

        print(f"\t😈😈 Welcome to the Learning Rate Sweep.  Because setting learning rate manually stinks😈😈")
        while lr >= min_lr:
            error = self.trial_error(setup, batch, lr)
            error_rising = self.rising_count(error, last_error, error_rising)
            last_error = error
            seq += 1
            self.log_sweep_trial(conn, run_id, seq, lr, error, error_rising)

            if error < best_error:
                best_error = error
                best_lr = lr

            if error_rising >= patience: break
            lr *= factor

        # ─── fine-tune around the coarse winner ─────────────────────────────
        coarse_lr = best_lr
        coarse_error = best_error
        if best_lr is not None and self.hyper.sweep_precision > 0:
            print(f"\t🏆 Coarse winner: LR = {coarse_lr:.2e} (error = {coarse_error:.5f})")
            print(f"\t🔬🔬🔬🔬🔬🔬\tCan fine tuning find better?\t🔬🔬🔬🔬🔬🔬")
            best_lr, best_error, seq = self.learning_rate_refine(setup, batch, run_id, best_lr, best_error, seq)
            self.refine_verdict(coarse_lr, coarse_error, best_lr, best_error)

        seq += 1
        msg = f"🏆🏆🏆 Best learning_rate = {best_lr:.2e} (best error = {best_error:.5f}) 🏆🏆🏆"
        print(f"\t😈\t{msg}\n")
        self.lr_log_trial(conn, run_id, seq, best_lr, best_error, "result", msg, is_best=1)
        return best_lr

    # NeuroEngine.py  --  NEW METHOD
    def trial_error(self, setup, batch, lr) -> float:
        """One 20-epoch probe at this LR. The single place a sweep trial gets scored."""
        setup["learning_rate"] = lr
        TRI = self.atomic_train_a_model(setup, RecordLevel.NONE, self.hyper, batch, epochs=20)
        return TRI.lowest_mae[TRI.checkpoint_phase]

    # NeuroEngine.py  --  NEW METHOD
    def rising_count(self, error, last_error, error_rising) -> int:
        """Consecutive rising trials. Trial 1 has no predecessor, so it cannot be a rise."""
        if last_error is None:      return 0
        if error > last_error:      return error_rising + 1
        return 0



    def log_sweep_trial(self, conn, run_id, seq, lr, error, error_rising):
        """Takes the counter as an argument, so the message cannot be built before it updates."""
        msg = f"😈\tLR:{lr:.1e} → err rising:{error_rising} Error:{error}"
        print(f"\t{msg}")
        self.lr_log_trial(conn, run_id, seq, lr, error, "test", msg)


    def refine_verdict(self, coarse_lr, coarse_error, best_lr, best_error):
        """Report whether fine tuning dethroned the coarse winner, and by how much."""
        if best_lr == coarse_lr:
            print(f"\t🔬 Verdict: coarse winner held its ground.")
        else:
            improved = (coarse_error - best_error) / coarse_error * 100
            print(
                f"\t🔬 Verdict: fine tuning wins!  LR {coarse_lr:.2e} → {best_lr:.2e}   error {coarse_error:.5f} → {best_error:.5f}  ({improved:.2f}% better)")

    def learning_rate_refine(self, setup: dict, batch, run_id, best_lr, best_error, seq):
        """
        Bisection in log10 space around the coarse-sweep winner.
        Each level: halve the step, test both shoulders, re-center on best of 3.
        hyper.sweep_precision = number of halving levels (0 = feature off).
        Cost: 2 trials per level.  Precision 4 lands within ~1.15x of the true bottom.
        """
        if best_lr is None: return best_lr, best_error, seq  # coarse sweep found nothing to refine
        conn = self.hyper.nf_db.conn

        center_exp = math.log10(best_lr)
        step = 0.5
        for level in range(self.hyper.sweep_precision):
            for exp in (center_exp - step, center_exp + step):  # both shoulders, always
                lr = 10 ** exp
                setup["learning_rate"] = lr
                TRI = self.atomic_train_a_model(setup, RecordLevel.NONE, self.hyper, batch, epochs=20)
                #error = TRI.lowest_mae
                error = TRI.lowest_mae[0]
                if error is None: error = float("inf")
                seq += 1
                msg = f"🔬\tLR:{lr:.2e} → level:{level + 1} Error:{error}  🔬"
                print(f"\t{msg}")
                self.lr_log_trial(conn, run_id, seq, lr, error, "refine", msg)
                if error < best_error:
                    best_error = error
                    best_lr = lr
            center_exp = math.log10(best_lr)  # re-center on champion (may not have moved)
            step *= 0.5
        return best_lr, best_error, seq

    # NeuroEngine.py  method: lr_log_clear  Update: complete method
    def lr_log_clear(self, conn, run_id):
        conn.execute("DELETE FROM log_lr_sweep WHERE run_id = ?", (run_id,))
        conn.commit()

    # NeuroEngine.py  method: lr_log_trial  Update: complete method
    def lr_log_trial(self, conn, run_id, seq, lr, mae, phase, message, is_best=0):
        conn.execute(
            "INSERT INTO log_lr_sweep (run_id, seq, learning_rate, mae, phase, is_best, message) VALUES (?,?,?,?,?,?,?)",
            (run_id, seq, lr, mae, phase, is_best, message)
        )
        conn.commit()

    def check_for_clear(self, batch):
        """This makes batch runs go in near linear time"""
        if batch.run_position % 10 == 0: #and batch.id_of_current > self.shared_hyper.nf_count:

            # Periodic RamDB reset for SUMMARY runs
            print(f"🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹\nGC COLLECT AT  {batch.run_position}\n🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹")
            # Memory profiling
            gc.collect()
            #process = psutil.Process()
            #mem_mb = process.memory_info().rss / 1024 / 1024
            #print(f"🧹 Run {batch.id_of_current}: Memory = {mem_mb:.1f} MB")





                                                          # NEW

    # NeuroEngine.py  method: clear_run_id  NEW: delete the prior archive, then negate
    def clear_run_id(self, run_id):
        """One generation only — the -N rows must go first or the negated keys collide.
           run_id negation is universal; batch_id negation is batch_history's own trick,
           so every batch-scoped query self-filters for free."""
        conn = self.hyper.nf_db.conn
        for table in self.tape_tables(conn):
            conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (-run_id,))
            conn.execute(f"UPDATE {table} SET run_id = ? WHERE run_id = ?", (-run_id, run_id))
        conn.execute("UPDATE batch_history SET batch_id = -batch_id WHERE run_id = ?", (-run_id,))
        conn.commit()

    # NeuroEngine.py  method: tape_tables  NEW: asked of the schema, so a tenth table joins by itself
    def tape_tables(self, conn):
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        return [r[0] for r in rows if self.records_by_run(conn, r[0])]

    # NeuroEngine.py  method: records_by_run  NEW
    def records_by_run(self, conn, table):
        return any(col[1] == "run_id" for col in conn.execute(f"PRAGMA table_info({table})"))

    # NeuroEngine.py  method: full_tape_runs  Update: targeted runs earn the tape, not the batch's lowest ids


