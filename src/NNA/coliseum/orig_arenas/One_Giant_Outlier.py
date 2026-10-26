import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class One_Giant_Outlier(BaseArena):
    """
    Regression task:
    All targets are clustered near 10, except for one massive outlier (e.g., 10,000).

    Designed to test:
        - MSE's sensitivity to large errors
        - MAE's robustness to outliers
        - Huber's compromise behavior
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
