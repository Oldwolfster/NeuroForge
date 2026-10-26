from typing import List, Tuple
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from src.NNA.engine.BaseArena import BaseArena


class Predict_HousingPrice__From_CaliforniaFeatures(BaseArena):
    """
    Real-world regression dataset from California housing data.
    Predicts median house value from 8 input features.
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

        return training_data, feature_names + ["MedianHouseValue"]
