import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Parity_Check(BaseArena):
    """
    Binary classification task:
    Given a binary input vector, predict whether it contains an odd number of 1s.

    Output: 1.0 for odd parity, 0.0 for even parity.
    """

    def __init__(self, num_samples: int, input_size: int = 5):
        self.num_samples = num_samples
        self.input_size = input_size

    def generate_training_data(self) -> Tuple[List[Tuple[float, ...]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            bits = [random.choice([0.0, 1.0]) for _ in range(self.input_size)]
            label = float(sum(bits) % 2)  # 1.0 if odd, 0.0 if even
            training_data.append((*bits, label))

        labels = [f"Bit {i+1}" for i in range(self.input_size)] + ["Parity (1=odd)"]
        return training_data, labels
