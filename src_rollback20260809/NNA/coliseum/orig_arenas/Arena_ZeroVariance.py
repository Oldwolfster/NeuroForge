from src.NNA.engine.BaseArena import BaseArena


class Arena_ZeroVariance(BaseArena):

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (10, 4, 100000),
            (10, 4, 100000),
            (10, 4, 100000),
        ]
        return samples, ["Years on Job", "Years College", "Salary"]
