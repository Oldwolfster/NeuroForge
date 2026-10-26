import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_ParityCheck(BaseArena):
    """
    Five bits, and whether an odd number of them are set. Flipping any single bit
    flips the answer, which is exactly what makes it hard.

    Lesson:       XOR generalized. No amount of width in a single layer will do it.
    Truth:        Parity = sum(bits) mod 2, five bits drawn independently
    Solvable by:  One hidden layer, minimum.
    Watch for:    Accuracy stuck at 50 percent without a hidden layer. That is the
                  expected result, not a bug to chase.
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
