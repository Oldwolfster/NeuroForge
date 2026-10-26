"""ATTIC -- not catalogued.

Not baseball. Byte-for-byte the credit-score arena with a baseball docstring bolted on; still returns Credit Score / Repaid?.
"""
from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class Baseball_simplest(BaseArena):
    """
    Purpose: Get into baseball
    1) randomly choose how many hits this player gets in a game.
    2) did the team win?
    3) For example, a score of 90 would normally repay, but there is a 10% chance it will not.
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
