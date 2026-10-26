import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_Linear_3X(BaseArena):
    """
    One input, one output, and a straight line through the origin. The smallest
    thing in the collection that can be trained.

    Lesson:       The first arena to open. One weight to find, and you already know
                  what it should come out to.
    Truth:        y = 3x + noise(sigma 1), x drawn from -10 to 10
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight walking to 3 and stopping there.
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
