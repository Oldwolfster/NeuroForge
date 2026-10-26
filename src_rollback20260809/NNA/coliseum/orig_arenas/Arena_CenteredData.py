from src.NNA.engine.BaseArena import BaseArena


class Arena_CenteredData(BaseArena):
    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (0, 0, 14000),
            (5, 0, 20000),
            (10, 0, 26000),
        ]
        return samples, ["Offset Feature", "Zero Feature", "Target"]

