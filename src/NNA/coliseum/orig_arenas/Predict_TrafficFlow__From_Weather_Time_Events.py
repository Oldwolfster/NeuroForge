from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
import math
class Predict_TrafficFlow__From_Weather_Time_Events(BaseArena):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, ...]]:
        training_data = []

        for _ in range(self.num_samples):
            time_of_day     = random.uniform(0, 24)
            day_of_week     = random.randint(0, 6)
            temperature     = random.uniform(-5, 35)
            precipitation   = random.uniform(0, 20)  # mm
            event_severity  = random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.08, 0.02])[0]

            # Encoding day as cyclical input
            day_sin = math.sin(2 * math.pi * day_of_week / 7)
            day_cos = math.cos(2 * math.pi * day_of_week / 7)

            # Simulate base traffic pattern
            base_flow = 500 - 50 * math.sin((time_of_day / 24) * 2 * math.pi)
            day_factor = 1 - 0.1 * day_cos
            weather_penalty = 1 - 0.005 * temperature - 0.03 * precipitation
            event_penalty = 1 - 0.1 * event_severity

            traffic_flow = base_flow * day_factor * weather_penalty * event_penalty
            traffic_flow += random.gauss(0, 10)  # noise

            training_data.append((
                time_of_day,
                day_sin,
                day_cos,
                temperature,
                precipitation,
                event_severity,
                traffic_flow
            ))

        return training_data, [
            "Time of Day", "Day Sin", "Day Cos", "Temp", "Precipitation", "Event Severity", "Vehicles per Minute"
        ]
