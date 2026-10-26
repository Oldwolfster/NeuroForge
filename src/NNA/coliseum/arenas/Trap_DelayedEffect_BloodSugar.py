import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Trap_DelayedEffect_BloodSugar(BaseArena):
    """
    Meal sizes and blood sugar readings, where a meal shows up in the bloodstream
    three readings later. Every row pairs a meal with a number it did not cause.

    Lesson:       The clearest memory trap in the collection. The cause is in the
                  data, just never on the same row as its effect.
    Truth:        Blood Sugar(t) = 80 + 0.5 x Meal Size(t-3) + noise(sigma 5)
    Solvable by:  Not an FFNN. A row-at-a-time model cannot reach backwards.
    Watch for:    Meal Size weight going to near zero, which is correct -- on its own
                  row it explains nothing.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        meals = [random.uniform(0, 100) for _ in range(self.num_samples + 5)]
        training_data = []
        for t in range(self.num_samples):
            # Blood sugar spikes 3 steps after a meal, not immediate
            delayed_effect = 0.5 * meals[t-3] if t >= 3 else 0
            sugar = 80 + delayed_effect + random.gauss(0, 5)
            training_data.append((meals[t], sugar))
        return training_data, ["Meal Size", "Blood Sugar"]