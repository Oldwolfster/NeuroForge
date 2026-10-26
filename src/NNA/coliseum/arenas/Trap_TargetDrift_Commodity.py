import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_TargetDrift_Commodity(BaseArena):
    """
    A price that depends on one input plus a baseline creeping upward forever. Time is
    given as a column, which is what makes this drift catchable.

    Lesson:       Non-stationary target with the clock supplied. Compare against the
                  drift arenas that hide it.
    Truth:        Price = 100 + cumulative drift + 2 x Demand + noise(sigma 2);
                  drift adds uniform(0, 0.1) per row
    Solvable by:  A single neuron, using Time.
    Watch for:    A positive weight on Time. Remove that column and the model is lost.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []
        drift = 0
        for t in range(self.num_samples):
            drift += random.uniform(0, 0.1)  # Slow upward drift
            base = 100 + drift
            x = random.uniform(0, 10)
            price = base + 2 * x + random.gauss(0, 2)
            training_data.append((t, x, price))
        return training_data, ["Time", "Demand", "Price"]