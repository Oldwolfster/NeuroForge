import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena
import math

class DiseaseRisk__From_HealthMetrics(BaseArena):
    """
    Predicts disease risk as a probability using synthetic health metrics.
    Output is a sigmoid of a centered, scaled linear function — values vary meaningfully from ~0 to ~1.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float, float, float]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            # Simulated health metrics
            bp    = random.uniform(90, 180)     # Blood Pressure
            sugar = random.uniform(70, 200)     # Blood Sugar
            bmi   = random.uniform(18, 40)      # Body Mass Index
            age   = random.uniform(18, 90)      # Age

            # Center features for stability
            bp_c    = bp - 120
            sugar_c = sugar - 100
            bmi_c   = bmi - 25
            age_c   = age - 50

            # Scaled linear function with noise
            linear = (
                0.05 * bp_c +
                0.07 * sugar_c +
                0.10 * bmi_c +
                0.03 * age_c +
                random.gauss(0, 0.5)  # mild noise
            )

            # Sigmoid to convert to probability in [0, 1]
            probability = 1 / (1 + math.exp(-linear))

            training_data.append((bp, sugar, bmi, age, probability))

        return training_data, ["Blood Pressure", "Blood Sugar", "BMI", "Age", "Disease Risk Probability"]
