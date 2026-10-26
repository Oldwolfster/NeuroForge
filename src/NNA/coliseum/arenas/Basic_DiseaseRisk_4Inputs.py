import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena
import math

class Basic_DiseaseRisk_4Inputs(BaseArena):
    """
    Four routine health readings and a risk score between 0 and 1. Each reading is
    centered on its normal value first, so the interesting rows are the abnormal ones.

    Lesson:       A sigmoid target over a linear combination, with the four inputs
                  on wildly different scales.
    Truth:        Risk = sigmoid(0.05 x (BP-120) + 0.07 x (Sugar-100)
                  + 0.10 x (BMI-25) + 0.03 x (Age-50) + noise(sigma 0.5))
    Solvable by:  A single neuron with a sigmoid output.
    Watch for:    Blood Sugar dominating once range is accounted for, despite BMI
                  carrying the larger coefficient.
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
