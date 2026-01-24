import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Predict_MedicalCost_WithOutliers(BaseArena):
    """
    Generates regression data predicting annual medical cost based on age and BMI.
    Introduces variable outliers controlled by outlier_factor.

    Cost = base_cost + (age_coeff * age) + (bmi_coeff * BMI) + noise + [rare outlier spike]
    """
    def __init__(self, num_samples: int, outlier_factor: float = 0.0):
        self.num_samples = num_samples
        self.outlier_factor = outlier_factor  # 0.0 = no outliers, 1.0 = full chance per sample

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            age         = random.uniform(18, 65)        # Age in years
            bmi         = random.uniform(18.5, 40.0)     # BMI index
            base_cost   = 3000
            age_coeff   = 200
            bmi_coeff   = 350
            noise       = random.gauss(0, 2500)

            cost = base_cost + (age_coeff * age) + (bmi_coeff * bmi) + noise

            # Introduce outlier by chance
            if random.random() < self.outlier_factor:
                cost += random.uniform(10000, 30000)  # Simulated rare expensive medical procedure

            training_data.append((age, bmi, cost))

        return training_data, ["Age", "BMI", "Annual Medical Cost ($)"]
