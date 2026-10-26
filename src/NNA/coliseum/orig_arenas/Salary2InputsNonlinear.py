import random
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class SalaryFromExperienceAndCollege(BaseArena):
    """
    Generates regression training data for a two-input neural network with a single target,
    introducing nonlinear dependencies between experience, college, and salary.

    Loss Function: Mean squared error (for the single target).

    Target = Base Salary + f(Years of Experience) + g(Years of College) + Noise
      where:
      - f(Years of Experience): Nonlinear growth with plateau
      - g(Years of College): Diminishing returns after 4 years
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        """
        Generates training data as tuples of (Years of Experience, College, Target).
        """
        training_data = []
        for _ in range(self.num_samples):
            # Input variables
            years_exp = random.uniform(0, 40)
            college = random.uniform(0, 8)

            # Parameters for Target (Salary Prediction)
            base_salary = 24000

            # Nonlinear growth for experience: exponential growth with plateau
            if years_exp < 10:
                coeff_exp = 2500
            elif years_exp < 20:
                coeff_exp = 1500
            else:
                coeff_exp = 500
            contribution_exp = coeff_exp * years_exp

            # Diminishing returns for college: strong impact up to 4 years, less after
            if college <= 4:
                coeff_col = 4000
            else:
                coeff_col = 2000
            contribution_col = coeff_col * college

            # Noise
            noise = random.gauss(0, 1000)  # Add noise to the target for realism

            # Calculate target
            target = base_salary + contribution_exp + contribution_col + noise

            # Append the tuple (inputs and outputs)
            training_data.append((years_exp, college, target))

        return training_data
