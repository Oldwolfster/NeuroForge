import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Delayed_Effect_BloodSugar(BaseArena):
    """
    The label depends on a delayed effect of the input.
    Only models that can encode memory/history will do well.
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