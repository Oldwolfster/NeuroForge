from src.NNA.engine.BaseArena import BaseArena


class Shape_CenteredData(BaseArena):
    """
    Three hand-written rows where one column is offset from zero and the other never
    moves. Built to be run at, not learned from.

    Lesson:       Scaler probe. Exposes anything that assumes a column has spread.
    Truth:        (0, 0, 14000), (5, 0, 20000), (10, 0, 26000)
    Solvable by:  Not a learning problem.
    Watch for:    The run completing. Zero Feature is the column that breaks things.
    """
    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (0, 0, 14000),
            (5, 0, 20000),
            (10, 0, 26000),
        ]
        return samples, ["Offset Feature", "Zero Feature", "Target"]

