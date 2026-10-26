import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_PiecewiseRegime(BaseArena):
    """
    Two inputs and a target that follows one rule on the left half of the range and a
    different one on the right. The two rules disagree about the sign of X.

    Lesson:       A conditional split where the condition is visible in the data.
                  Learnable, unlike the hidden-regime arenas next door.
    Truth:        Label = 3x + 2y + 5 when x < 10, otherwise -2x + 0.5y - 10
    Solvable by:  One hidden layer.
    Watch for:    A single neuron fitting a compromise slope that is wrong on both sides.
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