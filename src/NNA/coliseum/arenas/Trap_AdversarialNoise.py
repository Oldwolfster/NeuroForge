import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_AdversarialNoise(BaseArena):
    """
    A clean straight line with one row in ten replaced by a wild value. There is also
    a second input column that has never mattered.

    Lesson:       Contaminated labels. Ten percent bad rows against a relationship
                  that is otherwise exact.
    Truth:        Label = 2 x X + 3, except a 10 percent chance of uniform(-100, 100)
    Solvable by:  A single neuron, with a robust loss.
    Watch for:    MSE dragged far off the true line by the bad rows, while Y noise
                  holds zero weight throughout.
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