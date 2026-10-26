import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Hidden_Switch_Power(BaseArena):
    """
    The data-generating rule randomly switches between two formulas.
    No input feature reveals the switch.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []
        for _ in range(self.num_samples):
            switch = random.choice([0, 1])
            x = random.uniform(0, 10)
            if switch:
                y = 5 * x + 10
            else:
                y = -3 * x - 7
            y += random.gauss(0, 1)
            training_data.append((x, y))
        return training_data, ["Input", "Power Usage"]