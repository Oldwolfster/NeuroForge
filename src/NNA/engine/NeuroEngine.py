import random

from src.NNA.engine.TrainingBatch import TrainingBatch
from src.NNA.engine.Utils import set_seed


class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics
    def __init__(self, hyper):
        self.hyper                  = hyper
        self.training_data          = None
        self.run_a_batch()

    def run_a_batch(self):
        if self.hyper.resume_batch:     raise RuntimeError(f"Resume not yet implemented")
        else:                           batch = TrainingBatch(self.hyper)

