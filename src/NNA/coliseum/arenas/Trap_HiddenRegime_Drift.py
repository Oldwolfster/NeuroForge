import random
import math
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Trap_HiddenRegime_Drift(BaseArena):
    """
    Two input signals and a target, where the rule producing it flips every 150 rows
    and then flips back. Time is not one of the columns.

    Lesson:       Non-stationarity. The data looks learnable in any window you pick
                  and is not learnable overall.
    Truth:        Alternating every 150 rows -- 0.5x1 + 0.5x2 + a sine bump, or
                  1.5x1 - 0.7x2 + 20 minus that bump; noise sigma 3
    Solvable by:  Not an FFNN, absent a time column.
    Watch for:    The model settling on the average of two rules and being wrong in
                  both regimes rather than right in either.
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
