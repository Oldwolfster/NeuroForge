import random
import math
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


# Arena that truly demands memory or fails
class Regime_Trigger_Switch(BaseArena):
    """
    Brutally difficult arena that requires memory.

    There are two regimes (A and B) that *look identical in input* — the only way to
    predict the correct target is to know what regime you're in, which is toggled by
    a hidden input *several samples ago*.

    Without state, no network can track the flip — FFNNs will average the regimes and fail.

    Perfect for proving the limit of stateless feedforward networks.
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
arena = Regime_Trigger_Switch(5)
arena.generate_training_data()
