import random
import math
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


# Arena that truly demands memory or fails
class Trap_RegimeTrigger_Memory(BaseArena):
    """
    Two inputs that look identical in either regime, and a target whose rule toggles
    every seventeen rows. The toggle lives in the generator and never becomes a column.

    Lesson:       Proof that a stateless network has a ceiling. Two contradictory
                  answers per input and no way to tell which one applies.
    Truth:        Target = 2x1 + 3x2 or -2x1 + 1.5x2, alternating every 17 rows,
                  + noise(sigma 0.2)
    Solvable by:  Not an FFNN. This arena is here to fail.
    Watch for:    The learned weights averaging the two rules -- near zero on x1,
                  near 2.25 on x2.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self._regime_flag = False  # Internal switch that flips on hidden triggers

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        training_data = []
        trigger_interval = 17  # Every 17 samples, regime flips (but it's not an input)
        last_triggered = 0

        for i in range(self.num_samples):
            # Hidden regime toggle
            if (i - last_triggered) >= trigger_interval:
                self._regime_flag = not self._regime_flag
                last_triggered = i

            # Inputs are drawn from same range regardless of regime
            x1 = random.uniform(0, 1)
            x2 = random.uniform(0, 1)

            if self._regime_flag:
                y = 2 * x1 + 3 * x2 + random.gauss(0, 0.2)
            else:
                y = -2 * x1 + 1.5 * x2 + random.gauss(0, 0.2)

            training_data.append((x1, x2, y))

        return training_data, ["x1", "x2", "Target"]

# Generate preview
