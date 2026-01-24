import random
import math
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena

class Predict_EnergyOutput__From_Weather_Turbine(BaseArena):
    """
    Predicts energy output (kW) of a wind turbine given weather conditions and internal metrics.
    Introduces a cubic wind-speed relationship and non-linear interactions.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float, float, float, float]]:
        training_data = []

        for _ in range(self.num_samples):
            wind_speed      = random.uniform(0, 25)             # m/s
            temperature     = random.uniform(-10, 40)           # °C
            humidity        = random.uniform(20, 100)           # %
            turbine_rpm     = random.uniform(800, 1800)         # RPM
            blade_angle     = random.uniform(0, 45)             # degrees

            # Base power curve: wind speed cubed
            base_power = 0.5 * 1.225 * (wind_speed ** 3) * 0.0005

            # Efficiency drops at high RPM in high temps
            efficiency_penalty = 1 - 0.00005 * (turbine_rpm - 1200) * max(temperature - 20, 0)

            # Slight nonlinear bump from blade angle (e.g., stalls > 30°)
            angle_factor = 1 - 0.01 * max(blade_angle - 30, 0)

            # Humidity oscillation effect (periodic influence)
            humidity_wiggle = 1 + 0.05 * math.sin(humidity / 10)

            # Add it all together with noise
            output_kw = base_power * efficiency_penalty * angle_factor * humidity_wiggle
            output_kw += random.gauss(0, 0.05 * output_kw)

            training_data.append((wind_speed, temperature, humidity, turbine_rpm, blade_angle, output_kw))

        return training_data, [
            "Wind Speed (m/s)",
            "Temperature (C)",
            "Humidity (%)",
            "Turbine RPM",
            "Blade Angle (deg)",
            "Power Output (kW)"
        ]
