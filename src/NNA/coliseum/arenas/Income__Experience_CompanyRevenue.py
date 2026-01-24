import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Income__Experience_CompanyRevenue(BaseArena):
    """
    Generates regression training data with inputs of vastly different magnitudes.
    Demonstrates the issue of unnormalized inputs causing problems in training.
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
        return training_data
