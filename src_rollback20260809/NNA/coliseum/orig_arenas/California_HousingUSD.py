from typing import List, Tuple
from sklearn.datasets import fetch_california_housing
from src.NNA.engine.BaseArena import BaseArena


class CaliforniaHousingArena(BaseArena):
    """
    California Housing Prices Dataset

    Predicts the median house value (in dollars) for a census block group in California
    based on 8 numeric features:
      - MedInc: Median income (10k USD)
      - HouseAge: Median house age (years)
      - AveRooms: Avg rooms per household
      - AveBedrms: Avg bedrooms per household
      - Population: Block group population
      - AveOccup: Avg household occupancy
      - Latitude
      - Longitude

    Target values are scaled back to **real dollar amounts** (multiplied by 100,000).

    Source: California Housing dataset (1990 Census)
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
