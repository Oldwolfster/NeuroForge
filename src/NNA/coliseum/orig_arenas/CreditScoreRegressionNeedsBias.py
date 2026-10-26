from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class CreditScoreRegressionNeedsBias(BaseArena):
    """
    Concrete class that generates regression training data.
    It calculates a credit score between 0-100 and uses it to generate a continuous target value
    representing the repayment ratio (0.0 to 1.0).

    This version introduces a non-zero intercept (e.g., 0.5) to ensure the data doesn't pass through
    the origin. The repayment ratio is determined by adjusting the slope (score/200) and adding
    random noise for variance.
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
        return training_data
