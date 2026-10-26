
import random
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Basic_CustomerChurn_6Inputs(BaseArena):
    """
    SaaS subscribers scored from 0 to 1 on how likely they are to cancel next month.
    Two of the six columns are pure invention and feed nothing.

    Lesson:       Finding the four columns that matter among six. A gentler version
                  of the Red Herring trap.
    Truth:        Score = 0.6 x Invoice Lateness + 0.4 x (1 - Login Freq)
                  + 0.05 x Support Tickets + 0.01 x Days Since Login
                  + noise(sigma 0.02), clamped to 0..1
    Solvable by:  A single neuron.
    Watch for:    Unused 1 and Unused 2 driving their weights toward zero.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, ...]]:
        training_data = []
        for _ in range(self.num_samples):
            # Active features
            login_freq = random.uniform(0, 1)            # Normalized: 0 (rarely) to 1 (daily)
            support_tickets = random.randint(0, 10)      # Number of support issues
            days_since_login = random.uniform(0, 30)     # Higher means disengaged
            invoice_lateness = random.uniform(0, 1)      # 0 = on time, 1 = very late

            # Noisy or unused features (to simulate real-world messiness)
            unused1 = random.random()
            unused2 = random.random()
            # unused3 = random.random()
            # unused4 = random.random()
            # unused5 = random.random()
            # unused6 = random.random()

            # Synthetic rule for churn score
            churn_score = (
                0.6 * invoice_lateness +
                0.4 * (1 - login_freq) +
                0.05 * support_tickets +
                0.01 * days_since_login +
                random.gauss(0, 0.02)  # slight noise
            )
            churn_score = max(0, min(1, churn_score))  # Clamp to [0, 1]

            training_data.append((
                login_freq, support_tickets, days_since_login, invoice_lateness,
                unused1, unused2, churn_score
            ))

        return training_data, ["Login Freq", "Support Tickets", "Days Since Login", "Invoice Lateness", "Unused 1", "Unused 2", "Churn Risk"]