import random
import math
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Hidden_Regime_Shifter(BaseArena):
    """
    This arena is designed to appear learnable — but the function changes subtly over time.
    It's like a market with hidden 'moods' that flip the underlying logic.

    The signal is dependent, but not consistently — and without an explicit input indicating
    the mode, most models will average both regimes and fail.

    Only models that either memorize or find indirect predictors (e.g., phase or time)
    will break past ~5% accuracy unless told where the shift occurs.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        training_data = []

        for i in range(self.num_samples):
            # Inputs: two noisy signals, time step (implicitly tracking phase)
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 100)
            t = i

            # Hidden regime flips every 150 steps
            if (i // 150) % 2 == 0:
                # Regime A: sum-ish + bump
                y = 0.5 * x1 + 0.5 * x2 + math.sin(i / 10) * 10 + random.gauss(0, 3)
            else:
                # Regime B: difference-ish + offset + inverted bump
                y = 1.5 * x1 - 0.7 * x2 - math.sin(i / 10) * 10 + 20 + random.gauss(0, 3)

            training_data.append((x1, x2, y))

        return training_data, ["Input A", "Input B", "Target"]

# Preview 5 samples
arena = Hidden_Regime_Shifter(5)
arena.generate_training_data()
