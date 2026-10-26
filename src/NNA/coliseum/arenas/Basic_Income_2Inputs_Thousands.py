from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_Income_2Inputs_Thousands(BaseArena):
    """
    Salary that rises quickly at first and then flattens out, driven by experience
    alone. College is present as a column and contributes nothing.

    Lesson:       A saturating curve and a dead column, neither announced.
    Truth:        Salary = 45 + 35 x (1 - 1/(1 + 0.15 x Exp)) + noise(sigma 8),
                  in thousands. College is multiplied by zero.
    Solvable by:  One hidden layer.
    Watch for:    The College weight going to zero, and the fit flattening above
                  roughly 20 years.
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
        return training_data, ["Years Exp", "College", "Salary"]