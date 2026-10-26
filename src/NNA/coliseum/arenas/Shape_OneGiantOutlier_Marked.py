import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Shape_OneGiantOutlier_Marked(BaseArena):
    """
    The one-giant-outlier data with a second column stating outright which row is the
    monster -- the information a robust loss has to infer, handed over instead.

    Lesson:       Whether an explicit flag beats a robust loss function. Run this
                  against Shape_OneGiantOutlier and compare.
    Truth:        As Shape_OneGiantOutlier, with Outlier Marker set to 2 on the spike
                  and 1 on every other row
    Solvable by:  A single neuron.
    Watch for:    Whether MSE with the marker beats MAE without it.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []
        outlier_index = random.randint(0, self.num_samples - 1)

        for i in range(self.num_samples):
            x = random.uniform(0, 100)
            if i == outlier_index:
                y = 10_000.0
                z = 2  # Outlier marker
            else:
                y = 10 + random.uniform(-1.0, 1.0)
                z = 1  # Normal marker

            training_data.append((x, z, y))

        return training_data, ["Input X", "Outlier Marker", "Target Y"]
