import math
import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Trap_ChaoticSine(BaseArena):
    """
    A wave that oscillates faster the further you get from zero, because its frequency
    depends on x squared. No noise at all -- the difficulty is entirely in the shape.

    Lesson:       A function that is smooth everywhere and still changes character
                  completely across its range.
    Truth:        Target Y = sin(x) + sin(x^2), x drawn from -5 to 5
    Solvable by:  Several hidden layers, and it will still miss the fast ends.
    Watch for:    A good fit near zero and a mess at both extremes, where the
                  oscillation outruns whatever resolution the network has.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            x = random.uniform(-5, 5)
            y = math.sin(x) + math.sin(x ** 2)
            training_data.append((x, y))

        return training_data, ["Input X", "Target Y"]
