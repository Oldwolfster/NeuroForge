from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class Basic_CreditScore_Repaid(BaseArena):
    """
    A credit score from 1 to 100, and whether that loan came back. The score is not
    a hint about the answer -- it is the odds, so even a 90 defaults sometimes.

    Lesson:       Classification where the answer is genuinely uncertain. The noise
                  is the lesson, not a defect in the data.
    Truth:        P(Repaid) = Credit Score / 100
    Solvable by:  A single neuron.
    Watch for:    Accuracy topping out at 75 percent. That is the ceiling the coin
                  flips impose -- anything higher is memorization.
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            repayment = 1 if random.random() < (score / 100) else 0
            training_data.append((score, repayment))
        return training_data    , ["Credit Score", "Repaid?"], [ "Defaulted", "Paid It!"]
