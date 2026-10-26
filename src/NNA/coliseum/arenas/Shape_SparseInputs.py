import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Shape_SparseInputs(BaseArena):
    """
    Twenty columns, nearly all zero on any given row, and only three of them decide
    the label. Which three is drawn at random when the arena is constructed.

    Lesson:       Finding a handful of live columns in a wide, mostly empty matrix.
    Truth:        Label = 1 when the three secret columns sum above 0, otherwise 0
    Solvable by:  A single neuron.
    Watch for:    Three weights separating cleanly from seventeen that stay near zero.
    Notes:        The signal positions are chosen in __init__, so they can differ
                  between runs even at a fixed seed. See the review list.
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
