import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_Income_2Inputs_Linear(BaseArena):
    """
    Experience, years of college, and salary, with nothing hidden and nothing noisy.
    The one arena where the trained parameters should match the generator exactly.

    Lesson:       Ground truth recovery. You can read the answer off the model and
                  check it against the source.
    Truth:        Salary = 14000 + 12000 x Exp + 8000 x College
    Solvable by:  A single neuron.
    Watch for:    Bias 14000, weights 12000 and 8000. Anything else is an optimizer
                  problem, not a data problem.
    Notes:        Noise sigma is 0, so a perfect fit is genuinely available.
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
