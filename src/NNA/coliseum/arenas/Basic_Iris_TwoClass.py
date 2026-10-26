import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_Iris_TwoClass(BaseArena):
    """
    Two iris species told apart by four flower measurements. This is a synthetic
    stand-in drawn from clean per-species ranges, not the real Fisher measurements.

    Lesson:       The gentlest classification in the collection. The two species
                  barely overlap on any axis.
    Truth:        Setosa and Versicolor drawn from separate uniform ranges on all four
    Solvable by:  A single neuron.
    Watch for:    Petal Length alone nearly separating the classes.
    Notes:        Species is coded 0 for Setosa, 1 for Versicolor. Synthetic -- it
                  imitates the shape of the real data without being it.
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
            "Species"
        ]
