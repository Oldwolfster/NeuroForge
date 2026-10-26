from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_Income_2Inputs_Multiplicative(BaseArena):
    """
    Salary that scales with education rather than adding to it -- a year of college
    multiplies what experience already earned. Someone with neither earns very little.

    Lesson:       Multiplicative structure. Additive models miss the high corner
                  in a way that is systematic, not random.
    Truth:        Salary = (30 + 4 x Exp + noise(sigma 5)) x (College + 0.5)
    Solvable by:  One hidden layer.
    Watch for:    Underprediction concentrated where both inputs are high.
    Notes:        Same generator as Basic_Income_2Inputs_Engineered without the
                  pre-computed product column. Run the pair back to back.
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
            salary = salary * (college + .5)
            training_data.append((years_experience,college, salary))
            #training_data.append((years_experience, salary))
        return training_data, ["Years Exp", "College", "Salary"]