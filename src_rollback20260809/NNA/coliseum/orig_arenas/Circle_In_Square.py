import math
import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Circle_In_Square(BaseArena):
    """
    Nonlinear binary classification task:
    Predict whether a point lies inside a circle centered in a square.

    This tests the ability of models to learn curved decision boundaries.

    Label is:
        1.0 if inside the circle
        0.0 if outside
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        training_data = []

        center_x, center_y = 0.0, 0.0
        radius = 1.0

        for _ in range(self.num_samples):
            x = random.uniform(-1.5, 1.5)
            y = random.uniform(-1.5, 1.5)

            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            label = 1.0 if distance <= radius else 0.0

            training_data.append((x, y, label))

        return training_data, ["X Pos", "Y Pos", "Inside Circle"]
