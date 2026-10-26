import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Iris_Two_Class(BaseArena):
    """
    Binary classification task:
    Simulated subset of the Iris dataset with two classes:
        0.0 = Setosa
        1.0 = Versicolor

    Features:
        - Sepal Length (cm)
        - Sepal Width (cm)
        - Petal Length (cm)
        - Petal Width (cm)
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float, float, float]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            species = random.choice([0.0, 1.0])  # 0 = Setosa, 1 = Versicolor

            if species == 0.0:  # Setosa
                sepal_length = random.uniform(4.5, 5.8)
                sepal_width  = random.uniform(3.0, 4.5)
                petal_length = random.uniform(1.0, 1.9)
                petal_width  = random.uniform(0.1, 0.6)
            else:  # Versicolor
                sepal_length = random.uniform(5.0, 7.0)
                sepal_width  = random.uniform(2.0, 3.5)
                petal_length = random.uniform(3.0, 5.0)
                petal_width  = random.uniform(1.0, 1.8)

            training_data.append((sepal_length, sepal_width, petal_length, petal_width, species))

        return training_data, [
            "Sepal Length", "Sepal Width",
            "Petal Length", "Petal Width",
            "Species (0=Setosa, 1=Versicolor)"
        ]
