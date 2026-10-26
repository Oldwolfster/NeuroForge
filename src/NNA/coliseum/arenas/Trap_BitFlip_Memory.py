import random
from typing import List, Tuple, Optional
from src.NNA.engine.BaseArena import BaseArena


class Trap_BitFlip_Memory(BaseArena):
    """
    Eight bits in, and the same eight bits back out with one position inverted. Which
    position is supplied as an input, so the task is copy-with-one-change.

    Lesson:       Multi-output. The target here is eight columns wide, not one.
    Truth:        Output bits equal input bits, with the bit at flip_index inverted
    Solvable by:  A single neuron per output bit, given the index is handed over.
    Watch for:    Whether the engine handles a wide target at all. This is as much a
                  plumbing test as a learning one.
    Notes:        The class labels Not Flipped / Flipped do not fit an 8-column target.
                  See the review list.
    """
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
