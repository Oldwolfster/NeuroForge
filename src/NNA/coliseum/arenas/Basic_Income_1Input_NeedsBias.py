from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_Income_1Input_NeedsBias(BaseArena):
    """
    Salary from years on the job, with a floor under it -- a beginner does not start
    at zero. The cleanest arena in the collection for seeing what a bias term is for.

    Lesson:       Bias, isolated. Identical in shape to the no-bias income arenas
                  except that the line misses the origin.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 5), in thousands, floored at 0
    Solvable by:  A single neuron, but only one with a bias.
    Watch for:    The bias climbing toward 30. Disable it and the weight overshoots
                  trying to compensate.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            # Base salary of $30k plus $4k per year of experience
            salary = max(
                0.0,
                30 + (4 * years_experience) + random.gauss(0, 5)
            )
            training_data.append((years_experience, salary))
        return training_data, ["Years Exp", "Salary"]