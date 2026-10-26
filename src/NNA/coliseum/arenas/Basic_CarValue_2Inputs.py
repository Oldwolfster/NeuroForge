import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Basic_CarValue_2Inputs(BaseArena):
    """
    What a used car is worth, given how old it is and how far it has been driven.
    Both facts push the price down and neither one rescues the other.

    Lesson:         Two inputs, two negative weights. The clean step after a one-input line.
    Truth:          Car Value = 30000 - 1000 x Years - 50 x Miles(1000s) + noise(sigma 500)
    Loss Function:  Mean squared error.
    Solvable by:    A single neuron.
    Watch for:      Both weights landing negative and the bias near 30000.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            age_years    = random.uniform(0, 20)   # Cars 0–20 years old
            miles        = random.uniform(0, 200)  # 0–200k miles, in thousands
            base_value   = 30000                   # Brand new car base value
            coeff_age    = 1000                    # Loses $1k per year
            coeff_miles  = 50                      # Loses $50 per thousand miles
            noise        = random.gauss(0, 500)     # Some random noise (~$500 std dev)
            car_value    = (base_value
                            - (coeff_age * age_years)
                            - (coeff_miles * miles)
                            + noise)

            training_data.append((age_years, miles, car_value))

        # Labels: ["Age (Years)", "Miles (Thousands)", "Car Value ($)"]
        return training_data, ["Years", "Miles(1000s)", "Car Value"]
