import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_MultiModal_Temperature(BaseArena):
    """
    Temperature readings drawn from two separate populations, one clustered near 20
    degrees and one near 28. Hour of day is the only input and it tells you nothing.

    Lesson:       A two-humped target. The mean sits in the gap between the humps,
                  at a value that essentially never occurs.
    Truth:        Temperature = 20 + noise(sigma 1) or 28 + noise(sigma 2), chosen at
                  random; Hour of Day is independent of both
    Solvable by:  Nothing. The mode is not observable.
    Watch for:    The prediction landing near 24 and the loss refusing to improve.
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