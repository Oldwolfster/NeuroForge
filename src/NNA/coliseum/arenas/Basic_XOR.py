from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class Basic_XOR(BaseArena):
    """
    The four rows that broke the perceptron in 1969. Two bits in, and the answer is
    true when exactly one of them is.

    Lesson:       The reason hidden layers exist. Four rows, and a single neuron
                  cannot fit them.
    Truth:        Result = Input 1 XOR Input 2, all four combinations, no noise
    Solvable by:  One hidden layer with two neurons.
    Watch for:    A single neuron parking at 50 percent and staying there forever.
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[float, float]]:
        return ([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
                ["Input #1","Input #2", "Result"],["False","True"])
