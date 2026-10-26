import math
import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_CircleInSquare(BaseArena):
    """
    Points scattered on a square, labelled by whether they land inside a circle
    drawn in the middle. Nothing about the shape is given away -- only the two
    coordinates.

    Lesson:       The first curved boundary. No straight line separates these.
    Truth:        Inside = 1 when sqrt(X^2 + Y^2) <= 1, points drawn from a 3x3 square
    Solvable by:  One hidden layer.
    Watch for:    The corners. A network that is too small gets those wrong first.
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
