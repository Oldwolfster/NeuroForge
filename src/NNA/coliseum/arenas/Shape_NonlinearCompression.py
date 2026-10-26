from src.NNA.engine.BaseArena import BaseArena


class Shape_NonlinearCompression(BaseArena):
    """
    Four hand-written rows where one column jumps by powers of ten and the other never
    changes. Skew and zero variance sitting side by side.

    Lesson:       Scaler probe. Tests log and power transforms against a dead column
                  in the same pass.
    Truth:        (1, 1, 10000), (10, 1, 11000), (100, 1, 12000), (1000, 1, 13000)
    Solvable by:  Not a learning problem.
    Watch for:    A log transform turning the skewed column into an even ramp while
                  doing nothing at all to the constant.
    """

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
