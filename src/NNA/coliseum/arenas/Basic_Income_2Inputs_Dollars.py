from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_Income_2Inputs_Dollars(BaseArena):
    """
    Experience, years of college, and salary. College sits on the table as an input
    and contributes nothing -- its coefficient is multiplied away to zero.

    Lesson:       A dead column, in an arena whose name gives no warning about it.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 15). College is never used.
    Solvable by:  A single neuron with a bias.
    Watch for:    The College weight collapsing to zero. If it does not, the model
                  is fitting noise.
    Notes:        Named Dollars but generated in thousands -- the constants are 30 and 4.
                  Trap_Income_DeadInput is the same trap, labelled honestly.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            college = random.uniform(0, 8)

            # Base salary of $30k plus $4k per year of experience
            salary = max(
                0.0,
                30 + (4 * years_experience) + random.gauss(0, 15)
            )
             # Education contribution: diminishing returns after 4 years
            edu_contribution = 155 * (1 - 1/(1 + 0.5 * college))
            #This method has more impact on outcome
            edu_contribution = 0 * salary * college

            # Combined salary with some random variation
            salary = max(
                0.0,
                salary +  edu_contribution
            )

            training_data.append((years_experience, college, salary))

        return training_data, ["Years Exp", "College", "Salary"]