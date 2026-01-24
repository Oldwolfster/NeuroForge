import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Adversarial_Noise(BaseArena):
    """
    Most data is clean, but some samples are pure noise or flipped labels—
    exposes lack of robustness or overfitting.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for i in range(self.num_samples):
            x = random.uniform(0, 10)
            y = random.uniform(0, 10)
            # Standard: y = 2x + 3
            if random.random() < 0.1:
                # 10% adversarial/noise
                label = random.uniform(-100, 100)
            else:
                label = 2 * x + 3
            training_data.append((x, y, label))
        return training_data, ["X", "Y noise", "Label"]