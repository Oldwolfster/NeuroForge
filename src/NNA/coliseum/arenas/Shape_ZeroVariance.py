from src.NNA.engine.BaseArena import BaseArena


class Shape_ZeroVariance(BaseArena):
    """
    Three rows, all identical. Every column holds exactly one value, so nothing
    varies at all.

    Lesson:       Scaler probe. Anything that divides by spread meets a zero here.
    Truth:        Every row is (10, 4, 100000).
    Solvable by:  Not a learning problem.
    Watch for:    The run finishing at all. NaN or inf means a scaler guard has a hole.
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
