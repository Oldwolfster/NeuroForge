class Regularizer:
    def __repr__(self):
        """Custom representation for debugging."""
        return self.name

    def __init__(self, name, penalty_function, desc="", when_to_use="", best_for=""):
        """
        penalty_function: Callable(weight_array) -> scalar penalty (added to loss) AND
                          Callable(weight_array) -> gradient adjustment (added to weight gradient)
        """
        self.name = name
        self.penalty = penalty_function
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for

Regularizer_None = Regularizer(
    name="None",
    penalty_function=lambda weights: (0.0, [0.0] * len(weights)),
    desc="Disables regularization.",
    when_to_use="Debugging, baseline models.",
    best_for="Quick sanity checks."
)


def l2_penalty(weights, λ=0.01):
    penalty = λ * sum(w ** 2 for w in weights)
    gradient = [2 * λ * w for w in weights]
    return penalty, gradient

Regularizer_L2 = Regularizer(
    name="L2",
    penalty_function=l2_penalty,
    desc="Penalizes large weights (squared).",
    when_to_use="Helps prevent overfitting.",
    best_for="Most deep nets, especially with noisy data."
)


def l1_penalty(weights, λ=0.01):
    penalty = λ * sum(abs(w) for w in weights)
    gradient = [λ * (1 if w > 0 else -1 if w < 0 else 0) for w in weights]
    return penalty, gradient

Regularizer_L1 = Regularizer(
    name="L1 Lasso",
    penalty_function=l1_penalty,
    desc="Encourages sparsity by driving weights toward zero.",
    when_to_use="Feature selection or when weights should 'drop out'.",
    best_for="Sparse models, interpretable models."
)

