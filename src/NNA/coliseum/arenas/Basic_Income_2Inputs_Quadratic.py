import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Basic_Income_2Inputs_Quadratic(BaseArena):
    """
    Salary that peaks in mid-career and falls away on both sides, on top of the usual
    climb. Twenty years in is the top of the hill.

    Lesson:       A quadratic term. The single-neuron income arenas end here.
    Truth:        Salary = 14000 + 12000 x Exp + 8000 x College - 100 x (Exp - 20)^2
    Solvable by:  One hidden layer.
    Watch for:    A linear model cutting a straight chord through the hill, wrong at
                  both ends and wrong in the middle.
    Notes:        Noise sigma is 0. A gentler curvature constant is written and then
                  overwritten on the next line -- see the dead code list.
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
