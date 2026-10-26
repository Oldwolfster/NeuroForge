import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_Income_2Inputs_5Coefficients(BaseArena):
    """
    Salary from experience and years of college, where the two work together --
    a degree is worth more to someone who already has years behind them.

    Lesson:       An interaction term plus a mild quadratic. The two inputs stop
                  being independent.
    Truth:        Salary = 14000 + 10000 x Exp + 8000 x College
                  + 10 x Exp^2 + 2500 x Exp x College
    Solvable by:  One hidden layer, or a single neuron handed Exp x College.
    Watch for:    A linear fit failing worst where both inputs are high.
    Notes:        Noise sigma is 0, so the surface is exact and any error is model error.
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
