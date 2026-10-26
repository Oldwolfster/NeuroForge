import math
import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Chaotic_Function_Prediction(BaseArena):
    """
    Regression task:
    Predict the chaotic function y = sin(x) + sin(x²)

    This tests:
        - Nonlinear function fitting
        - Model smoothness vs overfitting
        - Tolerance to chaotic derivatives
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            x = random.uniform(-5, 5)
            y = math.sin(x) + math.sin(x ** 2)
            training_data.append((x, y))

        return training_data, ["Input X", "Target Y"]
