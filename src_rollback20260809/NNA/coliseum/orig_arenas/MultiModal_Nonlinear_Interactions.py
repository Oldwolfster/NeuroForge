import random
from typing import List, Tuple
from src.NNA.engine.BaseArena import BaseArena
import math

class MultiModal_Nonlinear_Interactions(BaseArena):
    """
    A brutally difficult regression problem with multiple challenging characteristics:
    1. Multi-modal output distribution (two distinct regimes)
    2. Complex nonlinear interactions between features
    3. Hidden threshold effects
    4. Multiplicative noise that scales with signal
    5. Feature interactions that only matter in certain regions
    6. Non-stationary relationships (coefficients change based on other features)

    This should expose limitations of simple 2-layer networks and require:
    - Deeper architectures to capture complex interactions
    - Better activation functions for smooth nonlinearities
    - Robust loss functions for multiplicative noise
    - Careful initialization for multi-modal learning
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        training_data = []

        for i in range(self.num_samples):
            # Generate 6 input features with different characteristics
            x1 = random.uniform(-2, 2)    # Primary driver
            x2 = random.uniform(0, 4)     # Interaction term
            x3 = random.uniform(-1, 3)    # Threshold activator
            x4 = random.uniform(-3, 1)    # Mode switcher
            x5 = random.uniform(0, 2)     # Multiplicative factor
            x6 = random.uniform(-1, 1)    # Noise modulator

            # Complex target function with multiple challenging properties

            # 1. Mode switching based on x4 (bimodal distribution)
            if x4 > -1.5:
                # Mode A: Complex polynomial interactions
                base_signal = (
                    15 * math.sin(x1 * math.pi) * math.exp(-x2/3) +
                    10 * x2**2 * math.tanh(x3) +
                    8 * x1 * x3 * math.cos(x2) if x3 > 0.5 else 0  # Threshold effect
                )

                # Multiplicative interactions (only active in Mode A)
                if x1 > 0 and x2 > 2:
                    base_signal += 20 * math.log(1 + x1 * x2) * (x5 + 0.1)

            else:
                # Mode B: Different functional form entirely
                base_signal = (
                    -12 * x1**3 +
                    18 * math.exp(-((x2-2)**2)/2) +  # Gaussian bump
                    -5 * abs(x3 - 1.5)**1.5 +        # V-shaped contribution
                    25 * (1 / (1 + math.exp(-5*(x1 + x2 - 2))))  # Sigmoid interaction
                )

            # 2. Non-stationary coefficients (relationships change based on context)
            context_modifier = math.tanh(x3 + x4/2)
            base_signal *= (1 + 0.4 * context_modifier)

            # 3. High-order interaction terms (require deeper networks)
            interaction_term = 0
            if abs(x1 + x2 - x3) < 1.0:  # Only in specific region
                interaction_term = 12 * x1 * x2 * x3 * math.sin(x4 * math.pi/2)

            # 4. Periodic component with varying frequency
            frequency = 1 + 2 * (x5 / 2)  # Frequency depends on x5
            periodic = 8 * math.sin(frequency * x1 * math.pi) * math.cos(x2)

            # 5. Multiplicative noise (harder to model than additive)
            signal_magnitude = abs(base_signal + interaction_term + periodic)
            noise_scale = 1 + 0.3 * signal_magnitude  # Noise scales with signal
            multiplicative_noise = random.gauss(1, 0.15 * x6**2)  # x6 modulates noise

            # 6. Final additive complexity
            additive_noise = random.gauss(0, noise_scale * 0.1)

            # Combine all components
            y = (base_signal + interaction_term + periodic) * multiplicative_noise + additive_noise

            # Add some extreme outliers to test robustness (2% chance)
            if random.random() < 0.02:
                y += random.choice([-1, 1]) * random.uniform(50, 100)

            training_data.append((x1, x2, x3, x4, x5, x6, y))

        return training_data, [
            "Primary_Driver",
            "Interaction_Base",
            "Threshold_Activator",
            "Mode_Switcher",
            "Multiplicative_Factor",
            "Noise_Modulator",
            "Complex_Output"
        ]

    def get_problem_characteristics(self):
        """
        Returns characteristics that should inform auto-ML decisions
        """
        return {
            "complexity_level": "very_high",
            "feature_interactions": "multiplicative_and_threshold",
            "noise_type": "multiplicative_and_additive",
            "distribution": "bimodal",
            "nonlinearity": "extreme",
            "suggested_min_depth": 4,
            "suggested_activation": "elu_or_swish",
            "suggested_loss": "huber_or_quantile",
            "outlier_robust": True,
            "requires_regularization": True
        }