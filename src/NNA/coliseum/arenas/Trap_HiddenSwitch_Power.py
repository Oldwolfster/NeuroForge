import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_HiddenSwitch_Power(BaseArena):
    """
    One input, one output, and a coin flipped invisibly on every row deciding which
    of two straight lines you are looking at. The lines slope opposite ways.

    Lesson:       The simplest unlearnable arena here. Two valid answers per input
                  and nothing whatsoever to tell them apart.
    Truth:        Power Usage = 5x + 10 or -3x - 7, chosen at random per row,
                  + noise(sigma 1)
    Solvable by:  Nothing. The switch is not observable.
    Watch for:    A flat line through the middle of two clouds. That is the correct
                  answer to an impossible question.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []
        for _ in range(self.num_samples):
            switch = random.choice([0, 1])
            x = random.uniform(0, 10)
            if switch:
                y = 5 * x + 10
            else:
                y = -3 * x - 7
            y += random.gauss(0, 1)
            training_data.append((x, y))
        return training_data, ["Input", "Power Usage"]