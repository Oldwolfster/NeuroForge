import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Basic_CustomerChurn_3Inputs(BaseArena):
    """
    Subscribers described by how long they stayed, what they paid, and how often
    they called support -- and whether they left. Almost all of them left.

    Lesson:       A badly imbalanced target. Accuracy looks excellent while the model
                  is doing nothing but answering "churned" every time.
    Truth:        P(Churn) = 0.2 + 0.1 x Tenure + 0.05 x Monthly Spend
                  + 0.15 x Service Calls, never capped at 1
    Solvable by:  A single neuron, though there is little here to learn.
    Watch for:    The churn rate in the data itself before you judge any model.
    Notes:        Monthly Spend alone drives the probability past 1 for most rows, so
                  roughly nine in ten churn. See the review list.
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
        return training_data, ["Tenure", "Monthly Spend", "Service Calls", "Churned"]