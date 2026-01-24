import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class AutoNormalize_Challenge(BaseArena):
    """
    Regression task:
    Inputs have wildly different magnitudes.
    Only some inputs matter.

    Purpose:
        - Tests whether smart defaults detect and normalize magnitude disparity
        - Validates optimizer stability under skewed features
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            # Key features (one tiny, one huge)
            tiny_signal  = random.uniform(0.01, 0.1)
            huge_signal  = random.uniform(1_000, 10_000)

            # Irrelevant distractors (medium-range)
            medium_noise1 = random.uniform(10, 20)
            medium_noise2 = random.uniform(50, 150)

            # Ground truth only uses tiny_signal and huge_signal
            y = 5 * tiny_signal + 0.001 * huge_signal + random.gauss(0, 0.5)

            training_data.append((tiny_signal, huge_signal, medium_noise1, medium_noise2, y))

        return training_data, [
            "Tiny Feature", "Huge Feature",
            "Medium Noise 1", "Medium Noise 2",
            "Target Y"
        ]
