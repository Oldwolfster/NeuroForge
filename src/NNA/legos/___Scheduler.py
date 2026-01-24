import math


class Scheduler:
    def __repr__(self):
        """Custom representation for debugging."""
        return self.name

    def __init__(self, name, get_lr_function, desc="", when_to_use="", best_for=""):
        self.name = name
        self.get_lr = get_lr_function  # (epoch, iter, base_lr, total_epochs) → new_lr
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for


Scheduler_Constant = Scheduler(
    name="Constant LR",
    get_lr_function=lambda epoch, iter, base_lr, total_epochs: base_lr,
    desc="Keeps learning rate fixed.",
    when_to_use="Debugging, manual tuning, stable tasks.",
    best_for="Quick experiments or stable plateaus."
)

Scheduler_StepDecay = Scheduler(
    name="Step Decay",
    get_lr_function=lambda epoch, iter, base_lr, total_epochs: base_lr * (0.5 ** (epoch // 10)),
    desc="Halves learning rate every 10 epochs.",
    when_to_use="Good for gradual convergence.",
    best_for="Training with plateaus or well-defined learning phases."
)

Scheduler_ExponentialDecay = Scheduler(
    name="Exponential Decay",
    get_lr_function=lambda epoch, iter, base_lr, total_epochs: base_lr * (0.95 ** epoch),
    desc="Multiplies learning rate by 0.95 every epoch.",
    when_to_use="General use when convergence slows gradually.",
    best_for="Training with long-term decay needs."
)

Scheduler_CosineAnnealing = Scheduler(
    name="Cosine Annealing",
    get_lr_function=lambda epoch, iter, base_lr, total_epochs: base_lr * 0.5 * (
                1 + math.cos(math.pi * epoch / total_epochs)),
    desc="Decays learning rate using a cosine curve.",
    when_to_use="Smooth convergence with a final cooldown.",
    best_for="Long training runs or fine-tuning."
)

Scheduler_OneCycle = Scheduler(
    name="OneCycle Policy",
    get_lr_function=lambda epoch, iter, base_lr, total_epochs: (
        base_lr * (1 + math.cos(math.pi * epoch / total_epochs))
        if epoch <= total_epochs // 2
        else base_lr * (1 - ((epoch - total_epochs // 2) / (total_epochs / 2)))
    ),
    desc="Rapid warmup followed by cooldown. Great for fast convergence.",
    when_to_use="You want fast convergence within a fixed number of epochs.",
    best_for="Smaller datasets or fast iteration."
)

Scheduler_WarmRestarts = Scheduler(
    name="Cosine with Warm Restarts",
    get_lr_function=lambda epoch, iter, base_lr, total_epochs: base_lr * 0.5 * (
                1 + math.cos(math.pi * (epoch % 10) / 10)),
    desc="Restarts the cosine cycle every 10 epochs.",
    when_to_use="Tasks prone to local minima or sharp descents.",
    best_for="Non-convex optimization landscapes."
)

# Optional: Bundle for auto-suggestion / UI dropdowns
all_schedulers = [
    Scheduler_Constant,
    Scheduler_StepDecay,
    Scheduler_ExponentialDecay,
    Scheduler_CosineAnnealing,
    Scheduler_OneCycle,
    Scheduler_WarmRestarts
]
