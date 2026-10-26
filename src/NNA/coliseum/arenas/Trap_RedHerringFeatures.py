import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Trap_RedHerringFeatures(BaseArena):
    """
    One input that determines the answer exactly, and two that are pure noise ranging
    twenty times wider. The target itself has no noise at all.

    Lesson:       Ignoring loud, irrelevant columns. The distractors dwarf the real
                  signal in magnitude.
    Truth:        Label = 4 x Signal - 7, exactly. Noise1 and Noise2 are never read.
    Solvable by:  A single neuron.
    Watch for:    Both noise weights driving to zero. A perfect fit is available here,
                  so anything short of one is distraction.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.uniform(0, 10)
            noise1 = random.uniform(-100, 100)
            noise2 = random.uniform(-100, 100)
            label = 4 * x - 7
            training_data.append((x, noise1, noise2, label))
        return training_data, ["Signal", "Noise1", "Noise2", "Label"]