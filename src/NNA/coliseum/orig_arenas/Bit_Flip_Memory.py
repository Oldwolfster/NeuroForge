import random
from typing import List, Tuple, Optional
from src.NNA.engine.BaseArena import BaseArena


class Arena_BitFlipMemory(BaseArena):
    def __init__(self, num_samples=100, bit_length=8):
        self.num_samples = num_samples
        self.bit_length = bit_length

    def generate_training_data(self)-> Tuple[List[Tuple[float, float, float, int]], List[str],Optional[List[str]]]:
        training_data = []

        for _ in range(self.num_samples):
            bits = [random.randint(0, 1) for _ in range(self.bit_length)]
            flip_index = random.randint(0, self.bit_length - 1)

            # Encode inputs: original bits + normalized flip index
            input_vector = bits + [flip_index / self.bit_length]

            # Create target: flip the bit at flip_index
            target_vector = bits[:]
            target_vector[flip_index] = 1 - target_vector[flip_index]  # flip

            # Append combined (input, target) sample
            training_data.append(tuple(input_vector + target_vector))

        # Input labels include flip index for clarity
        input_labels = [f"bit_{i}" for i in range(self.bit_length)] + ["flip_index"]
        target_labels = [f"bit_{i}_flipped" for i in range(self.bit_length)]

        return training_data, input_labels + target_labels, ["Not Flipped", "Flipped"]
