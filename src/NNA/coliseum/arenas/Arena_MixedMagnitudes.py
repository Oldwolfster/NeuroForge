from src.NNA.engine.BaseArena import BaseArena


class Arena_MixedMagnitudes(BaseArena):

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (0.1, 1000, 10000),
            (0.2, 1200, 12000),
            (0.3, 1100, 14000),
        ]
        return samples, ["Small Feature", "Large Feature", "Target"]
