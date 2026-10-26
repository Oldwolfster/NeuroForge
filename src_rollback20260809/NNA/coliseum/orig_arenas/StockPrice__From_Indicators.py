from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
import math

class Predict_StockPrice__From_Indicators(BaseArena):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, ...]]:
        training_data = []
        for _ in range(self.num_samples):
            mom3        = random.uniform(-3, 3)
            mom10       = random.uniform(-3, 3)
            volume_spike = random.uniform(0, 1)
            volatility  = random.uniform(0, 1)
            sentiment   = random.uniform(-1, 1)
            news_factor = random.choice([0, 0, 0.5, 1])  # 75% chance of no news impact

            # Compute adjusted price
            trend = 100 + 3 * mom3 + 1.5 * mom10
            if volume_spike > 0.6:
                trend += random.uniform(-8, 12)

            if sentiment > 0.5 and mom3 > 0:
                trend += 4 * sentiment

            noise = random.gauss(0, 3 + 1.5 * volatility)
            news_impact = news_factor * random.choice([-10, 10])

            adjusted_price = trend + noise + news_impact

            training_data.append((
                mom3, mom10, volume_spike, volatility, sentiment, news_factor, adjusted_price
            ))

        return training_data, [
            "Momentum (3d)", "Momentum (10d)", "Volume Spike", "Volatility", "Sentiment", "News Impact", "Adj Price"
        ]
