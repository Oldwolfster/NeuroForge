import random
from typing import List, Tuple, Optional
from src.NNA.engine.BaseArena import BaseArena
import math

class DefaultRisk__From_Income_Debt(BaseArena):
    """
    Predicts loan default risk based on Income, Debt, and engineered Debt-to-Income Ratio (DTI).
    Tests the model's ability to learn implicit relationships and benefit from engineered features.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float, int]], List[str],Optional[List[str]]]:
        training_data = []

        for _ in range(self.num_samples):
            income = random.uniform(20_000, 200_000)
            debt = random.uniform(1_000, 150_000)
            safe_income = max(income, 1e-3)
            dti_ratio = debt / safe_income

            # Default risk logic: risk increases sharply with higher DTI
            threshold = 0.4
            margin = 0.1
            if dti_ratio < threshold - margin:
                target = 0  # low risk
            elif dti_ratio > threshold + margin:
                target = 1  # high risk
            else:
                target = random.choice([0, 1])  # ambiguous zone

            training_data.append((income, debt, dti_ratio, target))

        return training_data, ["Income", "Debt", "Debt-to-Income Ratio", "Default Risk"], ["Low Risk","High Risk"]
