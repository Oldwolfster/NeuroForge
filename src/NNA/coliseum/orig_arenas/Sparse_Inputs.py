import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Sparse_Inputs(BaseArena):
    """
    Binary classification task with sparse input vectors:
    Most features are zero. Only a few positions influence the label.

    Useful for testing:
        - Model focus on sparse signals
        - Resilience to high-dimensional noise
    """

    def __init__(self, num_samples: int, input_size: int = 20):
        self.num_samples = num_samples
        self.input_size = input_size
        self.signal_indices = random.sample(range(self.input_size), 3)  # Only 3 positions matter

    def generate_training_data(self) -> Tuple[List[Tuple[float, ...]], List[str]]:
        training_data = []

        for _ in range(self.num_samples):
            inputs = [0.0] * self.input_size

            # Populate a few random non-zero entries
            for i in range(random.randint(2, 5)):
                idx = random.randint(0, self.input_size - 1)
                inputs[idx] = random.choice([1.0, -1.0])

            # Compute label based on signal indices (e.g., linear rule)
            signal_sum = sum(inputs[i] for i in self.signal_indices)
            label = 1.0 if signal_sum > 0 else 0.0

            training_data.append((*inputs, label))

        labels = [f"X{i+1}" for i in range(self.input_size)] + ["Target"]
        return training_data, labels
