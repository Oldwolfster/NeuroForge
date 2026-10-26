import random
import math
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Predict_Income_2_Inputs__HighlyNonlinear(BaseArena):
    """
    Very challenging nonlinear regression task that includes squared terms,
    interaction terms, and sine waves to introduce oscillatory behavior.

    This will definitely require at least 2 hidden layers to learn properly.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp = random.uniform(0, 40)
            college = random.uniform(0, 8)

            base_salary = 15000
            coeff_exp = 9000
            coeff_col = 7000
            coeff_sq_exp = 15
            coeff_interact = 1200
            coeff_sin_exp = 3000 # causes local minima
            coeff_sin_col = 1000 # causes local minima

            noise = random.gauss(0, 0)  # Disable noise for now

            salary = (
                base_salary
                + coeff_exp * years_exp
                + coeff_col * college
                + coeff_sq_exp * (years_exp ** 2)
                + coeff_interact * years_exp * college
                + coeff_sin_exp * math.sin(years_exp / 2.0)
                + coeff_sin_col * math.sin(college * 1.5)
                + noise
            )
            training_data.append((years_exp, college, salary))

        return training_data, ["Years on Job", "Years College", "Salary"]
