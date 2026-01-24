import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Predict_Income_2_Inputs__Nonlinear(BaseArena):
    """
    Nonlinearly separable regression problem using the same input structure (experience + college),
    but with interactions and curvature to make the function more complex.

    Salary = base_salary
             + coeff_exp * years_exp
             + coeff_col * college
             + coeff_sq_exp * (years_exp ** 2)
             + coeff_interact * years_exp * college
             + noise

    This version requires at least a small hidden layer to model properly.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp   = random.uniform(0, 40)
            college     = random.uniform(0, 8)

            # Base values
            base_salary     = 14000
            coeff_exp       = 10000
            coeff_col       = 8000
            coeff_sq_exp    = 10     # Small quadratic curve
            coeff_interact  = 2500    # Interaction between experience and college

            noise           = random.gauss(0, 0)  # Set to 0 now for clarity

            salary = (
                base_salary
                + coeff_exp * years_exp
                + coeff_col * college
                + coeff_sq_exp * (years_exp ** 2)
                + coeff_interact * years_exp * college
                + noise
            )
            training_data.append((years_exp, college, salary))

        return training_data, ["Years on Job", "Years College", "Salary"]
