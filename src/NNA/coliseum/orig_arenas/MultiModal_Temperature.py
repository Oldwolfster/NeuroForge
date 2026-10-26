import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class MultiModal_Temperature(BaseArena):
    """
    Target is drawn from one of two different regimes at random.
    Forces the model to learn a multi-modal distribution.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []
        for _ in range(self.num_samples):
            mode = random.choice([0, 1])
            if mode == 0:
                temp = 20 + random.gauss(0, 1)
            else:
                temp = 28 + random.gauss(0, 2)
            hour = random.uniform(0, 24)
            training_data.append((hour, temp))
        return training_data, ["Hour of Day", "Temperature"]