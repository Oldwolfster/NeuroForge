import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_Income_2Inputs_Engineered(BaseArena):
    """
    Experience and college, where salary is the product of the two rather than the
    sum -- and the product is handed over as a third column.

    Lesson:       Feature engineering. Multiplying two inputs is out of reach for one
                  neuron until you multiply them yourself.
    Truth:        Salary = (30 + 4 x Exp + noise(sigma 5)) x (College + 0.5)
    Solvable by:  A single neuron, thanks to the Exp x College column.
    Watch for:    Real weight on Exp x College. Drop that column and the same
                  architecture stops working -- see Basic_Income_2Inputs_Multiplicative.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            college = random.uniform(0, 8)
            # Original salary calculation with interaction
            base_salary = 30  # in thousands
            coeff_experience = 4  # in thousands per year
            noise = random.gauss(0, 5)

            intermediate_salary = base_salary + (coeff_experience * years_experience) + noise
            salary = intermediate_salary * (college + 0.5)

            # Compute the interaction term
            interaction = years_experience * college

            training_data.append((years_experience, college, interaction, salary))
        return training_data, ["Years Exp", "College", "Exp x College", "Salary"]