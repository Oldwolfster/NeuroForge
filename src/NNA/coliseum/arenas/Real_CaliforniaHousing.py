from typing import List, Tuple
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from src.NNA.engine.BaseArena import BaseArena


class Real_CaliforniaHousing(BaseArena):
    """
    Median house value for California census blocks, from eight facts about the
    neighbourhood. Features arrive already standardized.

    Lesson:       Real data with real correlation between columns, at a scale where
                  nothing has to be fixed before training.
    Source:       sklearn fetch_california_housing, 1990 US Census.
    Solvable by:  One hidden layer.
    Watch for:    Median Income carrying most of the weight, and the target flattening
                  against the census cap at 500,001 dollars.
    Notes:        Inputs are standard-scaled inside the arena, so a scaler on top of
                  that has nothing left to do.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, ...]], List[str]]:
        raw = fetch_california_housing()
        X = raw["data"]
        y = raw["target"]
        feature_names = raw["feature_names"]

        # Select subset
        X = X[:self.num_samples]
        y = y[:self.num_samples]

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Bundle features and target
        training_data = []
        for i in range(self.num_samples):
            features = tuple(X[i])
            target = float(y[i])
            training_data.append((*features, target))

        return training_data, feature_names + ["Median House Value"]
