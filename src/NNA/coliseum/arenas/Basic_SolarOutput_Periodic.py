import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena
import math

class Basic_SolarOutput_Periodic(BaseArena):
    """
    A solar panel's output across a year, high in summer and low in winter, with
    clouds knocking it down at random on about one day in ten.

    Lesson:       A cycle plus one-sided outliers. Cloud days can only subtract.
    Truth:        Output = 50 + 40 x sin(2 pi x Day / 365), minus uniform(0, 60) on
                  10 percent of days, plus noise(sigma 2)
    Solvable by:  One hidden layer, or a single neuron given sin and cos of the day.
    Watch for:    The fit sagging below the true curve. The outliers only pull one way.
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