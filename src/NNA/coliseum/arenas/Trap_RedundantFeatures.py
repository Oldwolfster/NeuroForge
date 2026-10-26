import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Trap_RedundantFeatures(BaseArena):
    """
    Two inputs that are almost the same number -- one is the other with a small jitter
    added -- and only the original drives the target.

    Lesson:       Collinearity. Two columns carrying the same information have no
                  unique correct weighting between them.
    Truth:        Redundant Copy = Signal + uniform(-0.2, 0.2);
                  Label = 2 x Signal + 3, exactly
    Solvable by:  A single neuron.
    Watch for:    The two weights summing to about 2 while splitting between
                  themselves differently on every run.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            true_signal = random.uniform(0, 10)
            noise = random.uniform(-0.2, 0.2)
            redundant = true_signal + noise
            label = 2 * true_signal + 3
            training_data.append((true_signal, redundant, label))
        return training_data, ["Signal", "Redundant Copy", "Label"]