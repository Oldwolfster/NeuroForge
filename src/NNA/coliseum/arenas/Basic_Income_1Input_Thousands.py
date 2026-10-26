from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_Income_1Input_Thousands(BaseArena):
    """
    Salary from years on the job, counted in thousands.

    Lesson:       Scale, isolated. This is the Dollars generator with every constant
                  divided by 1000 and nothing else touched.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 5), floored at 0
    Solvable by:  A single neuron with a bias.
    Watch for:    How much faster this converges than Basic_Income_1Input_Dollars,
                  given that no other thing differs.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            # Base salary of $30k plus $4k per year of experience
            salary = max(
                0.0,
                30 + (4 * years_experience) + random.gauss(0, 5)
            )
            college = random.uniform(0, 8)
            #salary = salary * (college + .5)
            #training_data.append((years_experience,college, salary))
            training_data.append((years_experience, salary))
        return training_data, ["Years Exp", "Salary"]