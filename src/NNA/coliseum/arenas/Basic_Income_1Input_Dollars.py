from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_Income_1Input_Dollars(BaseArena):
    """
    Salary from years on the job, in whole dollars -- the units a payroll system
    would actually use.

    Lesson:       Input and target magnitude. Nothing about the relationship changes,
                  but the distances an optimizer has to travel do.
    Truth:        Salary = 30000 + 4000 x Years Exp + noise(sigma 5000), floored at 0
    Solvable by:  A single neuron with a bias.
    Watch for:    Whether it trains at all without a scaler. Compare against
                  Basic_Income_1Input_Thousands, the same generator divided by 1000.
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
                30000 + (4000 * years_experience) + random.gauss(0, 5000)
            )
            college = random.uniform(0, 8)
            #salary = salary * (college + .5)
            #training_data.append((years_experience,college, salary))
            training_data.append((years_experience, salary))
        return training_data, ["Years Exp", "Salary"]
