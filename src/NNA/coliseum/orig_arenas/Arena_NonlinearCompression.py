from src.NNA.engine.BaseArena import BaseArena


class Arena_NonlinearCompression(BaseArena):

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples


    def generate_training_data(self):
        samples = [
            (1,    1, 10000),
            (10,   1, 11000),
            (100,  1, 12000),
            (1000, 1, 13000),
        ]
        return samples, ["Skewed Feature", "Constant", "Target"]
