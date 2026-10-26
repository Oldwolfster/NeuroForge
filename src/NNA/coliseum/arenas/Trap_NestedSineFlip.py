import random
import math
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_NestedSineFlip(BaseArena):
    """
    A wave that turns upside down for part of its range and then rights itself. Where
    you are on the axis is an input, so unlike most of its neighbours in this family
    this one is genuinely learnable.

    Lesson:       Architecture limits rather than impossibility. Watch a network that
                  is too small fail on purpose.
    Truth:        Signal = sin(3 x Position), negated for 3 <= Position < 6,
                  + noise(sigma 0.1)
    Solvable by:  Hidden layers, wider than you would guess.
    Watch for:    Error piling up at the two reversal points while the outer bands
                  fit cleanly.
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
        return training_data, ["Position", "Signal"]
