import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Piecewise_Regime(BaseArena):
    """
    For half the input space, one rule; for the other half, a different rule—
    requires the model to learn a conditional split.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.uniform(0, 20)
            y = random.uniform(0, 20)
            if x < 10:
                label = 3 * x + 2 * y + 5
            else:
                label = -2 * x + 0.5 * y - 10
            training_data.append((x, y, label))
        return training_data, ["X", "Y", "Label"]