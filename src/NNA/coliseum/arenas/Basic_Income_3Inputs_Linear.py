import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_Income_3Inputs_Linear(BaseArena):
    """
    Experience, college, and professional certifications against salary. Three
    inputs, all of them real, none of them interacting.

    Lesson:       Scaling up input count without adding difficulty. The step after
                  the two-input linear arena.
    Truth:        Salary = 24000 + 22000 x Exp + 2000 x College
                  + 1500 x Certifications + noise(sigma 2000)
    Solvable by:  A single neuron.
    Watch for:    All three weights recoverable, though Experience dwarfs the others.
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
        return training_data, ["Years Exp", "College", "Certifications", "Salary"]