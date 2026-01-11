from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class PutAnyNameHereOnlyFileMatters(BaseArena):
    """
    Purpose: Easily understandable synthetic test data.
    1) calculates a credit score between 0-100.
    2) It then uses the credit score as the percent chance the loan was repaid
    3) For example, a score of 90 would normally repay, but there is a 10% chance it will not.
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            repayment = 1 if random.random() < (score / 100) else 0
            #repayment = 0 if random.random() < (score / 100) else 1
            training_data.append((score, repayment))
        return training_data,     ["Credit Score", "Repaid?"], [ "Defaulted","Paid It!"]

