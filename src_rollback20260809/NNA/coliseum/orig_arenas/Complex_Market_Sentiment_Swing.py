import random
import math
from typing import List, Tuple

from src.NNA.engine.BaseArena import BaseArena


class Complex_Market_Sentiment_Swing(BaseArena):
    """
    Predicts consumer confidence index based on multiple partially redundant and periodically
    shifting signals: market volatility, news sentiment, unemployment claims, and public survey skew.

    There are correlations — but they flip over time, making it hard for the model to latch on.

    This tests whether the network can model features that are:
    - shifting in phase (temporal drift),
    - noisy,
    - partially redundant,
    - and interactively nonlinear.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float, float, float]], List[str]]:
        training_data = []

        for day in range(self.num_samples):
            t = day / 365.0  # normalize over years

            # Market volatility has a seasonal bump, plus noise
            volatility = 20 + 10 * math.sin(2 * math.pi * t) + random.gauss(0, 2)

            # News sentiment is inversely related — except it drifts every year
            sentiment = 50 + 10 * math.cos(2 * math.pi * t + random.uniform(-0.5, 0.5)) + random.gauss(0, 3)

            # Unemployment claims, noisy and has some correlation to sentiment
            unemployment = 200 + 20 * math.sin(4 * math.pi * t + 0.5) + random.gauss(0, 10)

            # Public opinion — survey responses with skew and occasional polarity flips
            if (day // 180) % 2 == 0:
                survey_skew = 50 + 15 * math.sin(2 * math.pi * t) + random.gauss(0, 5)
            else:
                survey_skew = 50 - 15 * math.sin(2 * math.pi * t) + random.gauss(0, 5)

            # Target is a weighted blend with drift and chaos
            confidence = (
                0.3 * volatility +
                0.2 * sentiment -
                0.25 * unemployment / 10 +
                0.25 * survey_skew +
                random.gauss(0, 2)
            )

            training_data.append((volatility, sentiment, unemployment, survey_skew, confidence))

        return training_data, ["Volatility", "Sentiment", "Unemployment Claims", "Survey Skew", "Consumer Confidence"]

# Generate a small preview of sample data
arena = Complex_Market_Sentiment_Swing(10)
sample_data, labels = arena.generate_training_data()
sample_data[:5], labels
