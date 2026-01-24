import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Predict_Income_2_Inputs(BaseArena):
    """
    Generates regression training data with a linear relationship between inputs and salary.
    This makes it solvable by a single neuron perceptron with two inputs.
    NOTE:   The Optimum bias and weights are the base salary and coefficients.
            When done training the parameters of the model should match what was set here!!!

    To make the salary calculation solvable by a single neuron perceptron with two inputs,
    we need to ensure that the relationship between the inputs and the salary is linear.

    Salary=Base Salary+(Coefficient1  ×Years of Experience)+(Coefficient2 × College)+Noise
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
            noise       = random.gauss(0, 0000)  # Add some noise for variance
            salary      = (base_salary + (coeff_exp * years_exp)
                                       + (coeff_col * college) + noise)
            training_data.append((years_exp, college, salary))
        return training_data, ["Years Exp","College","Salary"]
