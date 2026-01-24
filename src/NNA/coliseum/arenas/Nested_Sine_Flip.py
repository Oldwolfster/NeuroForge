import random
import math
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Nested_Sine_Flip(BaseArena):
    """
    Flipped sine wave segments:
    - Sin(3x) for x < 3
    - Flipped -Sin(3x) for 3 <= x < 6
    - Normal Sin(3x) again for x >= 6
    Same x-values lead to conflicting y-values based on region.
    Designed to test architecture limits (4-4-1 may underfit).
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.uniform(0, 9)
            if x < 3:
                y = math.sin(3 * x)
            elif x < 6:
                y = -math.sin(3 * x)
            else:
                y = math.sin(3 * x)
            y += random.gauss(0, 0.1)
            training_data.append((x, y))
        return training_data, ["x", "Signal"]
