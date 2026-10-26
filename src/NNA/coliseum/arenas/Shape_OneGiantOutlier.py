import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Shape_OneGiantOutlier(BaseArena):
    """
    Targets all sitting around 10, and exactly one at 10,000. The input has nothing
    to do with the target at all.

    Lesson:       Loss function comparison. MSE chases the monster, MAE ignores it,
                  Huber splits the difference.
    Truth:        Target Y = 10 + uniform(-1, 1), except one random row set to 10000
    Solvable by:  A single neuron predicting a constant, which is the right answer here.
    Watch for:    Where the prediction settles under each loss. That number is the
                  whole experiment.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        training_data = []

        # Choose random index for the outlier
        outlier_index = random.randint(0, self.num_samples - 1)

        for i in range(self.num_samples):
            x = random.uniform(0, 100)
            if i == outlier_index:
                y = 10_000.0  # The monster
            else:
                y = 10.0 + random.uniform(-1.0, 1.0)  # Near 10

            training_data.append((x, y))

        return training_data, ["Input X", "Target Y"]
