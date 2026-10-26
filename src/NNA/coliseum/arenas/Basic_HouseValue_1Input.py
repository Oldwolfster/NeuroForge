from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class Basic_HouseValue_1Input(BaseArena):
    """
    House price from square footage and nothing else. The noise is large enough that
    individual houses scatter widely around a line that is nevertheless exactly right.

    Lesson:       One input, one weight, heavy noise. Shows what a correct fit looks
                  like when the data itself is loose.
    Truth:        Price = 200 x Square Feet + noise(sigma 50000)
    Solvable by:  A single neuron. No bias needed.
    Watch for:    The weight settling near 200 while per-row error stays large.
                  Both of those are the right outcome.
    Notes:        Square footage runs 1500 to 4000, despite an old comment saying 500.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            # Generate square footage between 500 and 4000
            sqft = round(random.uniform(1500, 4000), 0)

            # Base price per square foot
            price_per_sqft = 200  # Adjust as needed

            # Calculate base price
            base_price = sqft * price_per_sqft

            # Try adding a flat amount like 20k and see if that makes the bias useful?

            # Add noise to the price
            noise = random.gauss(0, 50000)  # Mean 0, SD 50,000
            price = base_price + noise
            #price = base_price # temporarily remove the noise
            # Ensure price doesn't go below zero and round to nearest dollar
            #price = round(max(price, random.uniform(50000, 40000)),0)
            #price += 5000000 # I was expecting this to make use of bias
            training_data.append((sqft, price))
        return training_data, ["Square Feet", "Price"]