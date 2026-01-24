import math

from src.NNA.legos._LegoBase import LegoBase


class StrategyActivation(LegoBase):
    """Encapsulates an activation function and its derivative."""
    def __init__(self, function, derivative, bd_defaults, name="Custom"):
        super().__init__(name, ["hidden_activation", "output_activation"])

        self.function = function
        self.derivative = derivative
        self.bd_defaults = bd_defaults
        self.name = name

    def __call__(self, x):
        """Allows the object to be used as a function."""
        return self.function(x)


    def apply_derivative(self, x):
        """Compute the derivative for backpropagation."""
        return self.derivative(x)


Activation_ReLU         = StrategyActivation(
    function            = lambda x: x if x > 0 else 0,
    derivative          = lambda x: 1 if x > 0 else 0,
    bd_defaults         = [0, 1, 0.5],
    name                = "testReLU"
)

Activation_NoDamnFunction = StrategyActivation(
    function=lambda x: x,
    derivative=lambda x: 1,
    bd_defaults=[-1, 1, 0],
    name="test_linear"
)

