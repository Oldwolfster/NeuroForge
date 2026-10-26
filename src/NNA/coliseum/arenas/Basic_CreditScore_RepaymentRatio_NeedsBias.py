from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_CreditScore_RepaymentRatio_NeedsBias(BaseArena):
    """
    The repayment-ratio arena with a floor under it -- nobody pays back less than
    about half. The line no longer starts at the origin.

    Lesson:       Why a bias term exists. Identical to the plain RepaymentRatio
                  arena in every respect except the intercept.
    Truth:        Ratio = 0.5 + Credit Score / 200 + noise(sigma 0.1), clamped to 0..1
    Solvable by:  A single neuron, but only one with a bias.
    Watch for:    Bias near 0.5, weight near 0.005. Turn the bias off and watch it fail.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.uniform(0, 100)
            # Introduce a non-zero intercept (e.g., 0.5) and adjust the slope
            repayment_ratio = min(
                1.0,
                max(
                    0.0,
                    0.5 + (score / 200) + random.gauss(0, 0.1)
                )
            )
            training_data.append((score, repayment_ratio))
        return training_data, ["Credit Score", "Repayment Ratio"]