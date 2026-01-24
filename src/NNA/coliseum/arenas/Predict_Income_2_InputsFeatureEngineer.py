import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Salary2InputsLinear(BaseArena):
    """
    Generates regression training data with a linear relationship between inputs and salary.
    To enable the perceptron to model this non-linear relationship, we'll engineer
    a new feature that represents the interaction between years_experience and college.

    Interaction Term:
        Interaction Term = Years of Experience × CollegeInteraction Term=Years of Experience × College
        Expanded Salary Equation:
            The original salary calculation is: Salary = (30 + 4y) X (C +.4)
            When expanded (FOIL) it becomes Salary = (30C + 15) + (4y X C + 2y)
            This shows that the salary depends on:
                Years of Experience (𝑌)
                College (𝐶C)
                Interaction Term (Y×C)
                Constants
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
        return training_data