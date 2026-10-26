import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Custom_Function_Recovery(BaseArena):
    """
    Regression task:
    Predict y = 3x + noise

    This tests:
        - Ground truth recovery
        - Optimizer precision
        - Sensitivity to noise
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            x = random.uniform(-10, 10)
            noise = random.gauss(0, 1.0)  # Mild noise
            y = 3 * x + noise
            training_data.append((x, y))

        return training_data, ["Input X", "Target Y"]
