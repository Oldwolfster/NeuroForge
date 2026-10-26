import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Basic_FuelCost_2Inputs(BaseArena):
    """
    A trip's fuel bill from the distance driven and the price at the pump. The car's
    efficiency is fixed at 25 mpg and never appears as a column.

    Lesson:       A target built by multiplying two inputs rather than adding them.
    Truth:        Fuel Cost = (Miles / 25) x Gas Price + uniform(-1, 1)
    Solvable by:  One hidden layer. A single neuron cannot multiply two inputs.
    Watch for:    A linear fit landing on the average and missing both ends.
    Notes:        Fixed mpg, no traffic, no vehicle variation.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.mpg = 25  # constant fuel efficiency in miles per gallon

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        training_data = []
        for _ in range(self.num_samples):
            miles = random.uniform(0, 500)
            gas_price = random.uniform(2.5, 5.0)
            fuel_cost = (miles / self.mpg) * gas_price + random.uniform(-1.0, 1.0)
            training_data.append((miles, gas_price, fuel_cost))
        return training_data, ["Miles Driven", "Gas Price", "Fuel Cost"]
