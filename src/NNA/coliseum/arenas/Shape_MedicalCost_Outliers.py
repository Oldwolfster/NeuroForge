import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Shape_MedicalCost_Outliers(BaseArena):
    """
    Annual medical spend from age and BMI, with a dial on the class controlling how
    often a rare expensive procedure lands. The dial is off by default.

    Lesson:       Outlier dosage. The same clean problem, contaminated on demand.
    Truth:        Cost = 3000 + 200 x Age + 350 x BMI + noise(sigma 2500),
                  plus uniform(10000, 30000) at rate outlier_factor
    Solvable by:  A single neuron.
    Watch for:    How MSE moves as outlier_factor rises, against MAE or Huber on the
                  identical data.
    Notes:        outlier_factor defaults to 0.0, so out of the box this is clean.
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
