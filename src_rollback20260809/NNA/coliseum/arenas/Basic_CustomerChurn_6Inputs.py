
import random
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Basic_CustomerChurn_6Inputs(BaseArena):
    """
    🔮 Example Problem: Predicting Customer Churn Risk Score
    Imagine you’re building a system for a subscription-based SaaS company that wants to predict the risk of a customer churning — not as a yes/no (classification), but as a continuous score from 0 to 1. This score represents the likelihood that a customer will cancel their subscription in the next 30 days.

    🧠 Architecture
    Layer	Purpose
    🔢 Input	10–20 features: usage metrics, login frequency, support tickets, invoice payment timing, etc.
    🔁 Hidden #1	ReLU — learns low-level features: e.g., "low login + late invoice"
    🔁 Hidden #2	ReLU — combines patterns: "usage downtrend + payment friction"
    🔁 Hidden #3	ReLU — refines latent risk patterns
    🔚 Output	Sigmoid — outputs a smooth score between 0–1
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