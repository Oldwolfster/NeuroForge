import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Target_Drift_Commodity(BaseArena):
    """
    The target drifts upward over time, requiring the model to handle non-stationary data.
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
        return training_data, ["Time", "Feature", "Price"]