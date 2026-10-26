import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena
import math

class Pathological_Discontinuous_Chaos(BaseArena):
    """
    An extremely pathological regression problem designed to break simple architectures:
    1. Multiple discontinuities and regime switches
    2. Chaotic behavior in different regions
    3. Extreme sensitivity to input precision
    4. Non-smooth decision boundaries
    5. Heavy-tailed noise distribution
    6. Adversarial feature correlations
    7. Memory requirements (path-dependent behavior)

    This should require:
    - Very deep networks (6+ layers)
    - Advanced activation functions
    - Specialized loss functions
    - Careful regularization
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.history = []  # For path-dependent behavior

    def chaotic_map(self, x, chaos_param=3.9):
        """Logistic map - exhibits chaotic behavior"""
        return chaos_param * x * (1 - x)

    def fractal_function(self, x, y, iterations=5):
        """Simplified fractal-like function"""
        z_real, z_imag = x, y
        for i in range(iterations):
            new_real = z_real**2 - z_imag**2 + x
            new_imag = 2 * z_real * z_imag + y
            z_real, z_imag = new_real, new_imag
            if z_real**2 + z_imag**2 > 4:  # Divergence criterion
                return float(i) / iterations  # Normalized iteration count
        return 1.0  # Convergent

    def heavy_tailed_noise(self):
        """Cauchy distribution - heavy tails, infinite variance"""
        # Use Box-Muller to generate, then transform to Cauchy
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return math.tan(math.pi * (z / 6))  # Scale to make somewhat manageable

    def generate_training_data(self):
        training_data = []

        for i in range(self.num_samples):
            # Generate 8 features with different challenges
            x1 = random.uniform(-1, 1)      # Chaos driver
            x2 = random.uniform(-1, 1)      # Fractal input 1
            x3 = random.uniform(-1, 1)      # Fractal input 2
            x4 = random.uniform(-2, 2)      # Discontinuity trigger
            x5 = random.uniform(0, 1)       # Sensitivity amplifier
            x6 = random.uniform(-1, 1)      # Regime selector
            x7 = random.uniform(-1, 1)      # Correlation trap
            x8 = random.uniform(0, 1)       # Path dependency

            # Initialize complex target
            target = 0

            # 1. CHAOTIC REGION - sensitive to small changes
            if abs(x1) > 0.1:  # Avoid fixed point at 0
                chaos_input = (x1 + 1) / 2  # Map to [0,1] for chaotic map
                chaos_series = chaos_input
                for _ in range(10):  # Iterate chaotic map
                    chaos_series = self.chaotic_map(chaos_series)
                target += 50 * (chaos_series - 0.5)

            # 2. FRACTAL COMPONENT - requires complex boundary learning
            fractal_val = self.fractal_function(x2, x3)
            target += 30 * fractal_val * math.sin(10 * math.pi * fractal_val)

            # 3. MULTIPLE DISCONTINUITIES with different scales
            if x4 > 1.5:
                target += 100 * math.exp(x4 - 1.5)  # Exponential growth
            elif x4 > 0.5:
                target += -75 + 25 * (x4 - 0.5)**3  # Cubic region
            elif x4 > -0.5:
                target += 40 * math.sin(20 * math.pi * x4)  # High frequency oscillation
            elif x4 > -1.5:
                target += -60 * math.log(abs(x4 + 1.5) + 0.01)  # Logarithmic singularity
            else:
                target += 200 * (x4 + 2)**(-2)  # Inverse square law

            # 4. EXTREME SENSITIVITY - tiny changes, huge effects
            sensitivity_factor = 1000 * x5
            if abs(x1 - 0.333333) < 0.000001:  # Incredibly narrow spike
                target += sensitivity_factor

            # 5. REGIME SWITCHING with different functional forms
            if x6 > 0.3:
                # Regime A: Multiplicative chaos
                target *= (1 + 0.5 * math.sin(x7 * math.pi * x8))
            elif x6 > -0.3:
                # Regime B: Additive complexity with correlations
                correlation_trap = x7 * x8 * math.cos(x1 + x2 + x3)
                target += 80 * math.tanh(correlation_trap * 10)
            else:
                # Regime C: Path-dependent behavior
                self.history.append(x8)
                if len(self.history) > 5:
                    path_effect = sum(self.history[-5:]) / 5  # Moving average
                    target += 60 * path_effect * math.sin(math.pi * len(self.history) / 100)
                    self.history = self.history[-10:]  # Keep limited history

            # 6. ADVERSARIAL FEATURE CORRELATIONS
            # Create misleading correlations that don't actually predict target
            decoy_correlation = 0.3 * (x7**2 + x8**2) + random.gauss(0, 0.1)
            # This correlation is meaningless but might fool simple models

            # 7. HEAVY-TAILED NOISE - breaks MSE assumptions
            heavy_noise = self.heavy_tailed_noise()
            # Clip extreme outliers to keep somewhat tractable
            heavy_noise = max(-50, min(50, heavy_noise))
            target += heavy_noise

            # 8. FINAL PATHOLOGICAL TWIST - occasional complete randomness
            if random.random() < 0.05:  # 5% completely random outputs
                target = random.uniform(-200, 300)

            # Add the decoy correlation as a feature (red herring)
            training_data.append((x1, x2, x3, x4, x5, x6, x7, x8, decoy_correlation, target))

        return training_data, [
            "Chaos_Driver",
            "Fractal_X",
            "Fractal_Y",
            "Discontinuity_Trigger",
            "Sensitivity_Amplifier",
            "Regime_Selector",
            "Correlation_Trap_1",
            "Path_Dependency",
            "Decoy_Correlation",  # Red herring feature
            "Pathological_Target"
        ]

    def get_problem_characteristics(self):
        return {
            "complexity_level": "pathological",
            "discontinuities": "multiple_severe",
            "noise_type": "heavy_tailed_plus_outliers",
            "chaos_level": "extreme",
            "feature_interactions": "adversarial_and_misleading",
            "suggested_min_depth": 8,
            "suggested_activation": "multiple_types_needed",
            "suggested_loss": "quantile_or_robust",
            "requires_advanced_regularization": True,
            "path_dependent": True,
            "fractal_boundaries": True
        }