import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Shape_Income_MixedMagnitudes(BaseArena):
    """
    Salary from years of experience and the employer's annual revenue. One column
    tops out at 40 and the other at a hundred million.

    Lesson:       What unscaled magnitude disparity does to a gradient, on data that
                  reads like a real problem rather than a probe.
    Truth:        Salary = 30000 + 2000 x Years Exp + 0.0001 x Company Revenue
                  + noise(sigma 5000)
    Solvable by:  A single neuron, with scaling.
    Watch for:    Without a scaler, the Years Exp weight barely moving while Company
                  Revenue swings wildly.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp = random.uniform(0, 40)  # Small-scale feature
            company_revenue = random.uniform(1_000_000, 100_000_000)  # Large-scale feature
            base_salary = 30_000
            coeff_exp = 2000
            coeff_revenue = 0.0001  # Tiny weight for the large-scale feature
            noise = random.gauss(0, 5000)  # Add noise for realism
            salary = (base_salary +
                      coeff_exp * years_exp +
                      coeff_revenue * company_revenue +
                      noise)
            training_data.append((years_exp, company_revenue, salary))
        return training_data, ["Years Exp", "Company Revenue", "Salary"]