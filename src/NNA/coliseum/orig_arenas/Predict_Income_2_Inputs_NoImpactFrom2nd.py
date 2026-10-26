from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Salary2Inputs(BaseArena):
    """
    Concrete class that generates regression training data.
    It models years of experience (0 to 40) and uses it to generate a continuous target value
    representing the salary in thousands of dollars.

    This version introduces a non-zero intercept to ensure the data doesn't pass through
    the origin. The salary is determined by adjusting the slope and adding random noise for variance.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            # Base salary of $30k plus $4k per year of experience
            salary = max(
                0.0,
                30 + (4 * years_experience) + random.gauss(0, 5)
            )
            college = random.uniform(0, 8)
            #salary = salary * (college + .5)
            training_data.append((years_experience,college, salary))
            #training_data.append((years_experience, salary))
        return training_data
