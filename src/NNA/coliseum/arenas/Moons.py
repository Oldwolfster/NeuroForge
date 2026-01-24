from typing import List, Tuple
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from src.NNA.engine.BaseArena import BaseArena


class Predict_Binary__From_Moons(BaseArena):
    """
    Synthetic 2D classification dataset using interleaving half-circles.
    Ideal for testing nonlinear binary classification performance.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        X, y = make_moons(n_samples=self.num_samples, noise=0.2)
        X = StandardScaler().fit_transform(X)

        training_data = [(*tuple(x), float(label)) for x, label in zip(X, y)]

        return training_data, ["Input 1", "Input 2", "Answer"]
