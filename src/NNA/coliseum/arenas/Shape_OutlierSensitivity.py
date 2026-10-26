from src.NNA.engine.BaseArena import BaseArena


class Shape_OutlierSensitivity(BaseArena):
    """
    Four hand-written rows -- a clean ramp of three, and one sitting far outside the
    range of the others.

    Lesson:       Scaler probe. Min-Max squeezes the three good rows into a corner;
                  Robust should not.
    Truth:        (5, 1, 100000), (10, 2, 150000), (15, 3, 200000), (90, 8, 800000)
    Solvable by:  Not a learning problem.
    Watch for:    The scaled values of the first three rows. Under Min-Max they
                  collapse on top of each other.
    """

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
