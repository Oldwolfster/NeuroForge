# ═══ REPLACE FILE: Basic_HouseValue_Simple.py ════════════════════════════════
from src.NNA.engine.BaseArena import BaseArena
from typing import List, Tuple


class Basic_HouseValue_Simple(BaseArena):
    """
    House price from square footage and nothing else, with no noise at all. Three
    houses with round numbers, so every answer can be checked in your head.

    Lesson:       The first perceptron. A weight is a fact about the world: watch it
                  become the price per square foot.
    Truth:        Price = 200 x Square Feet, exactly
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight gliding toward 200 and arriving by the last epochs.
                  Then open Basic_HouseValue_1Input: same truth, added noise, and
                  the error no longer goes away.
    """
#🔶 NEW: three houses, round numbers: $200k, $280k, $360k
    HOUSES = (1000, 1400, 1800)
    NO_BIAS = True
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

#🔶 Update: fixed houses instead of random ones; cycles if asked for more than three
    def generate_training_data(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        houses        = [self.HOUSES[i % len(self.HOUSES)] for i in range(self.num_samples)]
        training_data = [(sqft, sqft * 200) for sqft in houses]
        return training_data, ["Sq Ft", "Price"]

