from src.NNA.engine.BaseArena import BaseArena


class Shape_ZeroVariance(BaseArena):
    """
    Scaler probe: every row is identical, so every feature has zero variance.
    Any scaler that divides by spread must survive this without exploding.
    """

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (10, 4, 100000),
            (10, 4, 100000),
            (10, 4, 100000),
        ]
        return samples, ["Years on Job", "Years College", "Salary"]
