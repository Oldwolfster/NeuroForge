import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Salary3InputsLinear(BaseArena):
    """
    Generates regression training data with a linear relationship between inputs and salary.
    This makes it solvable by a single neuron perceptron without adding extra inputs.
    NOTE:   Optimum bias should be the base salary!
            Optimum weights should be the coefficients.

    To make the salary calculation solvable by a single neuron perceptron with three inputs, we ensure that the relationship between the inputs and the salary is linear.

    Salary = Base Salary + (Coefficient1 × Years of Experience)
             + (Coefficient2 × College) + (Coefficient3 × Certifications) + Noise

    Loss Function: Mean squared error (since it's a regression task).
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp     = random.uniform(0, 40)  # Input 1
            college       = random.uniform(0, 8)   # Input 2
            certifications = random.uniform(0, 20)  # Input 3
            base_salary   = 24000
            coeff_exp     = 22000
            coeff_col     = 2000
            coeff_cert    = 1500
            noise         = random.gauss(0, 2000)  # Add some noise for variance
            salary        = (base_salary + (coeff_exp * years_exp)
                                           + (coeff_col * college)
                                           + (coeff_cert * certifications)
                                           + noise)
            training_data.append((years_exp, college, certifications, salary))
        return training_data
