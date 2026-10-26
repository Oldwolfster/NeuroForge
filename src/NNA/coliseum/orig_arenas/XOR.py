from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class XOR(BaseArena):
    """
    This class allows you to send literal training data
    for example if you need repeatable results
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[float, float]]:
        return ([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
                ["Input #1","Input #2", "Result"],["False","True"])
