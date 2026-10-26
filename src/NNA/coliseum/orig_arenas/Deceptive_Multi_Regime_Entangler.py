import random
import math
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


# Creating a devious arena
class Deceptive_Multi_Regime_Entangler(BaseArena):
    """
    Combines several learnable regimes — but:
    - The regime selector is *not* exposed as an input.
    - Some inputs have misleading correlations that flip per regime.
    - Signal strength varies drastically (some regimes weak, some strong).
    - There's one faint, reliable signal buried in the noise.

    This tests whether a network can disentangle complex, conflicting signal sources
    when no single rule applies globally — perfect for breaking naive generalization.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float, float, float]], List[str]]:
        training_data = []

        for i in range(self.num_samples):
            x1 = random.uniform(0, 1)  # weak linear
            x2 = random.uniform(0, 1)  # flips sign
            x3 = random.uniform(0, 1)  # noise
            x4 = random.uniform(0, 1)  # stable but weak signal

            regime = (i // 100) % 3  # 3 regimes cycling

            if regime == 0:
                # x1 and x2 agree positively, x3 is noise
                y = 10 * x1 + 5 * x2 + random.gauss(0, 1)
            elif regime == 1:
                # x2 is inverted, x3 is strong, x1 irrelevant
                y = -8 * x2 + 6 * x3 + random.gauss(0, 1)
            else:
                # only x4 matters, others are noise
                y = 20 * x4 + random.gauss(0, 1)

            training_data.append((x1, x2, x3, x4, y))

        return training_data, ["x1 (deceptive)", "x2 (inverts)", "x3 (noisy)", "x4 (weak true)", "Target"]

# Preview a few rows
#arena = Deceptive_Multi_Regime_En6tangler(5)
#arena.generate_training_data()
