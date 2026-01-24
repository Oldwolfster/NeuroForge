import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Predict_FuelCost__From_MilesDriven_GasPrice(BaseArena):
    """
    :description: Predicts total fuel cost from miles driven and gas price per gallon
    :intended_behavior: Tests a simple linear regression scenario across two inputs
    :label_summary: [Miles Driven (0–500), Gas Price ($/gal), Fuel Cost ($)]
    :notes: Assumes fixed MPG; no traffic or vehicle variability
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
