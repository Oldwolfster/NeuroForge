import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Red_Herring_Features(BaseArena):
    """
    Some features are pure noise—model must ignore them to do well.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.uniform(0, 10)
            noise1 = random.uniform(-100, 100)
            noise2 = random.uniform(-100, 100)
            label = 4 * x - 7
            training_data.append((x, noise1, noise2, label))
        return training_data, ["Signal", "Noise1", "Noise2", "Label"]