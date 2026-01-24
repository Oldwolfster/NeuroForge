import random
from src.ArenaSettings import HyperParameters
from src.NNA.engine.BatchCreator import BatchCreator
from src.NNA.engine.BatchRunner import BatchRunner
from src.NNA.engine.NNA_History import NNA_history
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo, RecordLevel
from src.NNA.engine.Utils import set_seed
from src.NNA.utils.db_prep_ram import create_weight_tables, prep_RamDB
from src.NNA.utils.dynamic_instantiate import instantiate_arena, dynamic_instantiate
#import psutil
import gc

from src.NeuroForge.NeuroForge import NeuroForge


class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics
    def __init__(self, hyper):
        self.hyper                          = hyper
        self.training_data                  = None
        self.hyper.db_ram                   = prep_RamDB()
        self.TRIs: list[TrainingRunInfo]    = []
        self.run_a_batch()
        #if self.TRIs:                       self.TRIs[0].db.copy_tables_to_permanent()
        if self.TRIs:                       NeuroForge(self.TRIs)

    def run_a_batch(self):
        if self.hyper.resume_batch:     raise RuntimeError(f"Resume not yet implemented")
        #else:                          batch_id = BatchCreator(self.hyper).create_a_batch()
        else:                           batch_id = BatchCreator(self.hyper).create_new_batch()
        batch =                         BatchRunner(batch_id, self.hyper)
        for run_id, setup in batch:
            print(f"NF - run id: {run_id} and batch_id: {batch_id}")
            print(f"\n💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪")
            print(f"💪 {batch.current_run} of {batch.total_runs}-{setup["arena"]} - these settings: {setup}")
            self.check_for_clear(batch)

            # Check if learning rate is specified in gladiator otherwise do a sweep.
            #if not setup.get("lr_specified", False):   setup["learning_rate"] = self.learning_rate_sweep(setup, batch)
            #else:                                      setup.pop("learning_rate", None)
            if not setup.get("lr_specified", False):
                setup["learning_rate"] = self.learning_rate_sweep(setup, batch)

            record_level        =  RecordLevel.FULL if batch.current_run <= self.hyper.nf_count else RecordLevel.SUMMARY
            TRI_latest          =  self.atomic_train_a_model(setup, record_level, self.hyper, batch, epochs=0) #epochs 0 is how it differentiates the 'real' run from a LR sweep
            if record_level     == RecordLevel.FULL: self.TRIs.append(TRI_latest)
            #NNA_history         . record_from_config(TRI_latest,  batch_id, run_id)
            NNA_history.record  (  TRI_latest, batch_id, run_id, batch.lego_mgr)

    def atomic_train_a_model(self,setup:dict,record_level, hyper:HyperParameters, batch: BatchRunner, epochs):  # ATAM is short for  -->atomic_train_a_model
        set_seed                ( setup.get("seed"))
        training_data           = instantiate_arena(setup["arena"], hyper.training_set_size)  # Passed to base_gladiator through TRI
        set_seed                ( setup.get("seed"))                             # Reset seed as it likely was used in training_data
        create_weight_tables    ( hyper.db_ram, batch.current_run)
        TRI                     = TrainingRunInfo   ( hyper, training_data, setup, record_level,batch.current_run)
        NN                      = dynamic_instantiate(setup["gladiator"], 1, "coliseum/gladiators", TRI)
        NN.train                ( epochs)  # Actually train model
        return                  TRI

    def learning_rate_sweep(self, setup: dict,  batch) -> float:
        """
        Sweep learning rates from 1.0 down to 1e-12 (logarithmically).
        Stops early if no improvement after `patience` trials.
        Modifies only 'learning_rate' in the setup dict.
        """
        gladiator    = setup.get("gladiator")
        start_lr     = 1e-6
        min_lr       = 1e-14
        max_lr       = 1e3
        factor       = 10

        # ─── NEW: early‐stop patience setup ───────────────────────────
        patience          = 3     # number of consecutive no‐improve steps before quitting
        no_improve_count  = 0
        # ───────────────────────────────────────────────────────────────

        best_error   = float("inf")
        best_lr      = None
        lr           = start_lr
        trials       = 0

        print(f"\t😈😈 Welcome to the Learning Rate Sweep.  Because setting learning rate manually stinks😈😈")
        #while lr >= min_lr and lr < max_lr and trials < max_trials:
        while lr >= min_lr:
            setup["learning_rate"] = lr
            TRI = self.atomic_train_a_model(setup, RecordLevel.NONE, self.hyper, batch, epochs=20)
            error = TRI.lowest_mae
            print(f"\t😈\tLR:{lr:.1e} → No Improve:{no_improve_count} Error:{error}  ")

            # TODO make this on the way up
            # ─── check for gradient explosion ───────────────────────────
            if (error is None or error > 1e20) and factor == 10:           # gradient explosion – no need to search higher
                factor              = 0.1       # reverse direction
                lr                  = start_lr
                no_improve_count    = 0         # Reset patience once we flip direction
                continue

            # ─── track best_error and reset or increment patience ──────
            if error is not None and error < best_error:
                    best_error          = error
                    best_lr             = lr
                    no_improve_count    = 0
            else:   no_improve_count    += 1

            if no_improve_count >= patience and factor < 1 : break
            lr *= factor
            if lr > max_lr:
                no_improve_count    = 0 # reset for downward search(ignoerd in upward)
                factor              = .1
                lr                  = start_lr * factor
            trials += 1

        print(f"\t😈\t🏆🏆🏆 Best learning_rate = {best_lr:.1e} (best error = {best_error:.5f} 🏆🏆🏆)\n")
        return best_lr


    def learning_rate_sweep_garbage(self, setup, batch):
        """
        Bidirectional sweep: test upward from 1e-6 to 1.0, then downward from 1e-6 to 1e-15.
        Stops early if no improvement after `patience` trials in each phase.
        Returns the best learning rate found.
        """
        start_lr = 1e-6
        min_lr = 1e-15
        max_lr = 1.0
        max_trials = 20
        patience = 3

        best_error = float("inf")
        best_lr = None
        trials = 0
        gradient_explosions = 0

        print(f"\t😈😈 Welcome to the Learning Rate Sweep. Because setting learning rate manually stinks 😈😈")

        # ╔══ Phase 1: Upward sweep (1e-6 → 1.0) ══╗
        lr = start_lr
        factor = 10
        no_improve_count = 0

        while lr <= max_lr and trials < max_trials:
            setup["learning_rate"] = lr
            TRI = self.atomic_train_a_model(setup, RecordLevel.NONE, self.hyper, batch, epochs=20)
            error = TRI.last_mae
            print(f"\t😈\tLR:{lr:.1e} → Error:{error}")
            # print(f"Conv={TRI.converge_cond}")
            trials += 1

            # Check for gradient explosion - skip this LR
            if TRI.converge_cond == "Gradient Explosion":
                gradient_explosions += 1
                break  # Stop upward phase on first explosion

            # Track best (only if not exploded)
            if error < best_error:
                best_error = error
                best_lr = lr
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

            lr *= factor

        # ╔══ Phase 2: Downward sweep (1e-6 → 1e-15) ══╗
        lr = start_lr
        factor = 0.1
        no_improve_count = 0  # Reset patience for phase 2

        while lr >= min_lr and trials < max_trials:
            setup["learning_rate"] = lr
            TRI = self.atomic_train_a_model(setup, RecordLevel.NONE, self.hyper, batch, epochs=20)
            error = TRI.last_mae
            print(f"\t😈\tLR:{lr:.1e} → Error:{error}")
            #print(f"Conv={TRI.converge_cond}")
            trials += 1

            # Check for gradient explosion - skip this LR
            if TRI.converge_cond == "Gradient Explosion":
                gradient_explosions += 1
                break  # Stop downward phase on first explosion

            # Track best (only if not exploded)
            if error < best_error:
                best_error = error
                best_lr = lr
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

            lr *= factor

        # Raise error if all LRs caused gradient explosion
        if best_lr is None:
            raise RuntimeError(
                f"Learning rate sweep failed: All {trials} learning rates tested caused gradient explosion. "
                f"This likely indicates a problem with the model architecture, initialization, or data scaling."
            )

        print(f"\t😈\t🏆🏆🏆 Best learning_rate = {best_lr:.1e} (best error = {best_error:.5f}) 🏆🏆🏆\n")
        return best_lr




    def check_for_clear(self, batch):
        """This makes batch runs go in near linear time"""
        if batch.current_run % 10 == 0: #and batch.id_of_current > self.shared_hyper.nf_count:

            # Periodic RamDB reset for SUMMARY runs
            self.hyper.db_ram = prep_RamDB()
            print(f"🧹 Reset RamDB at run {batch.current_run}")
            # Memory profiling
            gc.collect()
            #process = psutil.Process()
            #mem_mb = process.memory_info().rss / 1024 / 1024
            #print(f"🧹 Run {batch.id_of_current}: Memory = {mem_mb:.1f} MB")
