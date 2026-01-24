import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class One_Giant_OutlierExplainable(BaseArena):
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
