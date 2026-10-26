import random
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Basic_Income_2Inputs_Piecewise(BaseArena):
    """
    A career with three distinct phases -- fast early growth, a slower middle, and a
    plateau. The boundaries between them are hard edges, not gradual transitions.

    Lesson:       Sharp regime changes with no column marking where they happen.
    Truth:        Exp contributes 10000/yr below 10, 5000/yr to 30, then 2000/yr;
                  plus 8000 x College; base 20000; noise sigma 1000
    Solvable by:  One hidden layer, wide enough to place two kinks.
    Watch for:    The elbows at 10 and 30 years. A smooth fit rounds them off.
    Notes:        A rare-spike branch fires at probability 0.000005, which is
                  effectively never. An old comment calls it 5 percent.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp = random.uniform(0, 40)
            college = random.uniform(0, 8)

            # Piecewise behavior for experience
            if years_exp < 10:
                exp_component = years_exp * 10000  # Fast growth early
            elif years_exp < 30:
                exp_component = 100000 + (years_exp - 10) * 5000  # Slower growth
            else:
                exp_component = 200000 + (years_exp - 30) * 2000  # Plateau effect

            col_component = college * 8000  # Linear for college
            base_salary = 20000
            if random.random() < 0.000005:  # 5% chance
                noise = random.gauss(500000, 100000)  # Very bad spike
            else:
                noise = random.gauss(0, 1000)  # Normal noise


            salary = base_salary + exp_component + col_component + noise
            training_data.append((years_exp, college, salary))

        return training_data, ["Years Exp", "College", "Salary"]