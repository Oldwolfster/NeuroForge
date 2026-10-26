from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple

class Predict_Hits__From_AB_H_HR_BB_Past4Games(BaseArena):
    """
    Predicts hits in the next game using stats from the previous 4 games.
    Each input sample is a 4x4 matrix: [AB, H, HR, BB] for 4 games.
    The target is the hits in game 5.
    """

    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[list, float]]:
        training_data = []
        for _ in range(self.num_samples):
            sequence = []
            for _ in range(4):  # last 4 games
                ab = random.randint(2, 6)
                h  = random.randint(0, ab)
                hr = random.randint(0, min(h, 2))
                bb = random.randint(0, 3)
                sequence.append([ab, h, hr, bb])

            # Game 5 (tomorrow)
            ab5 = random.randint(2, 6)
            h5  = random.randint(0, ab5)
            target = h5

            training_data.append((sequence, target))

        return training_data
