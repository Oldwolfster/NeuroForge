import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena
import math

class Chaotic_Solar_Periodic(BaseArena):
    """
    Energy output with a strong periodic (sinusoidal) trend and random chaotic dips (clouds).
    Tests ability to model cycles and handle outliers.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []
        for day in range(self.num_samples):
            time = day % 365
            base = 50 + 40 * math.sin(2 * math.pi * time / 365)  # yearly cycle
            chaos = -random.uniform(0, 60) if random.random() < 0.1 else 0  # 10% chance of cloud
            output = base + chaos + random.gauss(0, 2)
            training_data.append((time, output))
        return training_data, ["Day of Year", "Energy Output"]