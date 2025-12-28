import random
from src.ArenaSettings import HyperParameters
from src.NNA.engine.BatchCreator import BatchCreator
from src.NNA.engine.BatchRunner import BatchRunner
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo, RecordLevel
from src.NNA.engine.Utils import set_seed
from src.NNA.utils.db_prep import create_weight_tables, prep_RamDB
from src.NNA.utils.dynamic_instantiate import instantiate_arena, dynamic_instantiate
#import psutil
import gc

class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics
    def __init__(self, hyper):
        self.hyper                  = hyper
        self.training_data          = None
        self.run_a_batch()

    def run_a_batch(self):
        if self.hyper.resume_batch:     raise RuntimeError(f"Resume not yet implemented")
        else:                           batch_id = BatchCreator(self.hyper).create_a_batch()

        TRIs: list[TrainingRunInfo]     = []
        batch                           = BatchRunner(batch_id=batch_id, db_dsk=self.hyper.db_dsk)
        for setup in batch:
            print(f"\n💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪")
            print(f"💪 {batch.current_run} of {batch.total_runs}-{setup["arena"]} - these settings: {setup}")
            self.check_for_clear(batch)

            # Check if learning rate is specified in gladiator otherwise do a sweep.
            if not setup.get("lr_specified", False):   setup["learning_rate"] = self.learning_rate_sweep(setup)
            else:                                      setup.pop("learning_rate", None)

            record_level = RecordLevel.FULL if batch.current_run <= self.hyper.nf_count else RecordLevel.SUMMARY
            TRI_latest = self.atomic_train_a_model(setup, record_level, self.hyper, batch, epochs=0) #epochs 0 is how it differentiates the 'real' run from a LR sweep
            if record_level == RecordLevel.FULL: TRIs.append(TRI_latest)

    def atomic_train_a_model(self,setup:dict,record_level, hyper:HyperParameters, batch: BatchRunner, epochs):  # ATAM is short for  -->atomic_train_a_model
        seed_to_use = setup.get("seed") # guranteed to have seed
        set_seed(seed_to_use)
        training_data = instantiate_arena(setup["arena"], hyper.training_set_size)  # Passed to base_gladiator through TRI
        set_seed(seed_to_use)  # Reset seed as it likely was used in training_data
        create_weight_tables(hyper.db_ram, batch.current_run)
        TRI = TrainingRunInfo(hyper, training_data, setup, record_level)

        # Below temporarily removed to test training data generation.
        NN = dynamic_instantiate(setup["gladiator"], 1, "coliseum/gladiators", TRI)
        #NN.train(epochs)  # Actually train model
        #record_results(TRI, batch_id, run_id)  # Store Config for this model #TODO make use of RecordLevel
        #return TRI

    def learning_rate_sweep(self, setup):
        pass

    def check_for_clear(self, batch):
        if batch.current_run % 10 == 0: #and batch.id_of_current > self.shared_hyper.nf_count:
            # Periodic RamDB reset for SUMMARY runs
            self.hyper.db_ram = prep_RamDB()
            print(f"🧹 Reset RamDB at run {batch.current_run}")
            # Memory profiling
            gc.collect()
            #process = psutil.Process()
            #mem_mb = process.memory_info().rss / 1024 / 1024
            #print(f"🧹 Run {batch.id_of_current}: Memory = {mem_mb:.1f} MB")
