from src.NNA.engine.BaseArena import BaseArena


class Arena_OutlierSensitivity(BaseArena):

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (5, 1, 100000),
            (10, 2, 150000),
            (15, 3, 200000),
            (90, 8, 800000),  # Outlier!
        ]
        return samples, ["Years on Job", "Years College", "Salary"]
