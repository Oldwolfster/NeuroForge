from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Basic_CreditScore_RepaymentRatio(BaseArena):
    """
    A credit score and the fraction of the loan that came back, as a number rather
    than a yes or no. The same story as the Repaid arena, asked as a regression.

    Lesson:       One dataset posed two ways. Run both and compare what you learn.
    Truth:        Ratio = Credit Score / 100 + noise(sigma 0.1), clamped to 0..1
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight settling near 0.01 and the bias staying near 0.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.uniform(0, 100)
            repayment_ratio = min(1.0, max(0.0, (score / 100)
                            + random.gauss(0, 0.1)))
            training_data.append((score, repayment_ratio))
        return training_data, ["Credit Score", "Repayment Ratio"]