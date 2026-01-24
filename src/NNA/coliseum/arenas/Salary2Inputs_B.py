from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Salary2Inputs_B(BaseArena):
    """
    Concrete class that generates regression training data.
    It models years of experience (0 to 40) and years of college (0 to 8)
    to generate a continuous target value representing the salary in thousands of dollars.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            college = random.uniform(0, 8)

            # Base salary starts at $45k
            base_salary = 45

            # Experience contribution: steeper early growth that tapers off
            exp_contribution = 35 * (1 - 1/(1 + 0.15 * years_experience))

            # Education contribution: diminishing returns after 4 years
            #edu_contribution = 15 * (1 - 1/(1 + 0.5 * college))
            edu_contribution = 0 * (1 - 1/(1 + 0.5 * college))

            # Combined salary with some random variation
            salary = max(
                0.0,
                base_salary + exp_contribution + edu_contribution + random.gauss(0, 8)
            )

            training_data.append((years_experience, college, salary))
        return training_data