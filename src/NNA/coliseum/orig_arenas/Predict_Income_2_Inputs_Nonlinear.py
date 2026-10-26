import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Predict_Income_2_Inputs_Nonlinear(BaseArena):
    """
    A nonlinear extension of the simple salary prediction arena.

    Introduces a quadratic relationship with years of experience,
    making it solvable only by a model with at least one hidden layer.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp   = random.uniform(0, 40)
            college     = random.uniform(0, 8)
            base_salary = 14000
            coeff_exp   = 12000
            coeff_col   = 8000
            # 🔁 Quadratic bump simulating "peak earning years" (e.g., mid-career)
            nonlinear_component = -100 * (years_exp - 20) ** 2 / 400  # Smooth hill centered at 20 years
            nonlinear_component = -40000 * (years_exp - 20) ** 2 / 400

            salary = (
                base_salary +
                (coeff_exp * years_exp) +
                (coeff_col * college) +
                nonlinear_component +
                random.gauss(0, 0)  # Add noise if desired
            )
            training_data.append((years_exp, college, salary))
        return training_data, ["Years on Job", "Years College", "Salary"]
