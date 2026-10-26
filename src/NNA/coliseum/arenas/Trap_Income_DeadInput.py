from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Trap_Income_DeadInput(BaseArena):
    """
    Experience, years of college, and salary, where college sits on the table and does
    nothing. The column name says so out loud.

    Lesson:       A dead input, labelled honestly. Confirms the model can zero a
                  column it has no use for.
    Truth:        Salary = 30 + 4 x Years Exp + noise(sigma 5). College is never read.
    Solvable by:  A single neuron with a bias.
    Watch for:    The College weight going to zero and staying. Compare
                  Basic_Income_2Inputs_Dollars -- same trap, no warning in the label.
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
            training_data.append((years_experience,college, salary))
            #training_data.append((years_experience, salary))
        return training_data, ["Years Exp", "College (no effect)", "Salary"]