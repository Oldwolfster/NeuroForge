from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Salary2Inputs_C(BaseArena):
    """
    Concrete class that generates regression training data.
    It models years of experience (0 to 40) and uses it to generate a continuous target value
    representing the salary in thousands of dollars.

    This version introduces a non-zero intercept to ensure the data doesn't pass through
    the origin. The salary is determined by adjusting the slope and adding random noise for variance.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            college = random.uniform(0, 8)

            # Base salary of $30k plus $4k per year of experience
            salary = max(
                0.0,
                30 + (4 * years_experience) + random.gauss(0, 15)
            )
             # Education contribution: diminishing returns after 4 years
            edu_contribution = 155 * (1 - 1/(1 + 0.5 * college))
            #This method has more impact on outcome
            edu_contribution = 0 * salary * college

            # Combined salary with some random variation
            salary = max(
                0.0,
                salary +  edu_contribution
            )

            training_data.append((years_experience, college, salary))

        return training_data
