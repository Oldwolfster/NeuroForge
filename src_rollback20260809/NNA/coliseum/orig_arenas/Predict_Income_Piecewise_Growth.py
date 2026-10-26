import random
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Salary2InputsPiecewise(BaseArena):
    """
    Generates regression training data with a single target,
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
        training_data = []
        for _ in range(self.num_samples):
            years_exp = random.uniform(0, 40)
            college = random.uniform(0, 8)

            # Piecewise behavior for experience
            if years_exp < 10:
                exp_component = years_exp * 10000  # Fast growth early
            elif years_exp < 30:
                exp_component = 100000 + (years_exp - 10) * 5000  # Slower growth
            else:
                exp_component = 200000 + (years_exp - 30) * 2000  # Plateau effect

            col_component = college * 8000  # Linear for college
            base_salary = 20000
            if random.random() < 0.000005:  # 5% chance
                noise = random.gauss(500000, 100000)  # Very bad spike
            else:
                noise = random.gauss(0, 1000)  # Normal noise


            salary = base_salary + exp_component + col_component + noise
            training_data.append((years_exp, college, salary))

        return training_data

