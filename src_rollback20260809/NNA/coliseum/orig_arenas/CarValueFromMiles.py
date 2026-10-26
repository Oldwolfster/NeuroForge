import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena


class Predict_Car_Value_From_Age_Miles(BaseArena):
    """
    Generates regression training data simulating car value based on:
    - Age (in years)
    - Miles driven (in thousands)

    True formula (linear):
    Car Value = Base Value - (Age Coefficient × Age) - (Miles Coefficient × Miles) + Noise

    This is intentionally very solvable by a single neuron with two inputs (age, miles).

    Loss Function: Mean Squared Error.
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
        return training_data, ["Years", "Miles(1000s)", "Value"]
