from typing import List, Tuple
from sklearn.datasets import fetch_california_housing
from src.NNA.engine.BaseArena import BaseArena


class Real_CaliforniaHousing_USD(BaseArena):
    """
    The same census blocks as the standardized version, priced in actual dollars with
    the eight features left raw. Population runs to the thousands while average
    bedrooms hovers near 1.

    Lesson:       What unscaled real data does to training. This is the arena that
                  makes the case for scalers.
    Source:       sklearn fetch_california_housing, 1990 US Census; target x 100,000.
    Solvable by:  One hidden layer, once the inputs are scaled.
    Watch for:    Divergence without a scaler. Compare Real_CaliforniaHousing, which
                  is this same data pre-standardized.
    """

    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, ...]], List[str]]:
        """
        Returns:
            - List of tuples: (inputs..., target)
            - List of labels: matching order of tuple values
        """
        data = fetch_california_housing()
        features = data.data[:self.num_samples]
        targets = data.target[:self.num_samples] * 100_000  # Convert to USD

        training_data = [(*feature_row, target) for feature_row, target in zip(features, targets)]

        labels = [
            "Median Income (10k)",
            "House Age",
            "Avg Rooms",
            "Avg Bedrooms",
            "Population",
            "Avg Occupancy",
            "Latitude",
            "Longitude",
            "Median House Value (USD)"
        ]

        return training_data, labels
