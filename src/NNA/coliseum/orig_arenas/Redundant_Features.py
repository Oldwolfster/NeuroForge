import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Redundant_Features(BaseArena):
    """
    Two highly correlated features, but only one is predictive.
    Models must avoid overfitting to noise in the redundant feature.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            true_signal = random.uniform(0, 10)
            noise = random.uniform(-0.2, 0.2)
            redundant = true_signal + noise
            label = 2 * true_signal + 3
            training_data.append((true_signal, redundant, label))
        return training_data, ["Signal", "Redundant Copy", "Label"]