import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class CustomerChurnArena(BaseArena):
    """
    Generates binary classification data for customer churn prediction.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float, int]]:
        training_data = []
        for _ in range(self.num_samples):
            tenure = random.uniform(0, 10)  # Years
            monthly_spend = random.uniform(10, 100)
            customer_service_calls = random.randint(0, 5)

            # Determine churn probability based on features
            churn_probability = 0.2 + 0.1 * tenure + 0.05 * monthly_spend + 0.15 * customer_service_calls

            # Randomly determine churn based on probability
            if random.random() < churn_probability:
                churn = 1  # Churned
            else:
                churn = 0  # Not churned

            training_data.append((tenure, monthly_spend, customer_service_calls, churn))
        return training_data