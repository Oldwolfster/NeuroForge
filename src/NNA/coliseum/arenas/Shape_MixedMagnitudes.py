from src.NNA.engine.BaseArena import BaseArena


class Shape_MixedMagnitudes(BaseArena):
    """
    Three hand-written rows, one column near 0.1 and the other near 1000. The minimal
    version of the scale problem.

    Lesson:       Scaler probe. Unscaled, the large column swamps the gradient entirely.
    Truth:        (0.1, 1000, 10000), (0.2, 1200, 12000), (0.3, 1100, 14000)
    Solvable by:  Not a learning problem.
    Watch for:    What each scaler does to the small column. Three rows is few enough
                  to check by hand.
    """

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self):
        samples = [
            (0.1, 1000, 10000),
            (0.2, 1200, 12000),
            (0.3, 1100, 14000),
        ]
        return samples, ["Small Feature", "Large Feature", "Target"]
