class BlameSignal:
    """
    💥 Replaces traditional loss functions.
    Focuses on the signal (gradient) used to adjust weights — not the loss value.

    Attributes:
        name: Human-readable name of the blame signal
        short_name: Optional short label for compact display
        description: What this signal represents
        gradient: Function to compute the signal (gradient of error)
        optional_value: Optional function to compute the loss value for reporting
        legacy_label: (Optional) Traditional name this maps from (e.g., "MSE", "MAE")
    """

    def __init__(
        self,
        gradient,
        name="Custom Signal",
        short_name="Custom",
        description="",
        optional_value=None,
        legacy_label=None
    ):
        self.gradient = gradient
        self.name = name
        self.short_name = short_name
        self.description = description
        self.optional_value = optional_value
        self.legacy_label = legacy_label

    def compute_blame(self, y_pred, y_true):
        return self.gradient(y_pred, y_true)

    def compute_optional_value(self, y_pred, y_true):
        if self.optional_value is None:
            return None
        return self.optional_value(y_pred, y_true)

    def __call__(self, y_pred, y_true):
        return self.compute_blame(y_pred, y_true)

    def __repr__(self):
        return f"BlameSignal '{self.name}'"

TrueErrorSignal = BlameSignal(
    gradient=lambda pred, target: pred - target,
    name="True Error Signal",
    short_name="TrueError",
    description="Uses raw error as blame. The most honest signal.",
    optional_value=lambda pred, target: np.abs(pred - target)  # Optional
)


DirectionalOnlyError = BlameSignal(
    gradient=lambda pred, target: np.sign(pred - target),
    name="Directional Only Error",
    short_name="±1",
    description="Applies +1 or -1 blame. Ignores magnitude completely.",
    legacy_label="MAE"
)
SquaredErrorSignal = BlameSignal(
    gradient=lambda pred, target: 2 * (pred - target) / _get_n(target),
    name="Squared Error Signal",
    short_name="2e",
    description="Standard MSE-style gradient. Blame grows quadratically.",
    legacy_label="MSE"
)
LogitsBCE_Signal = BlameSignal(
    gradient=lambda logit, target: 1 / (1 + np.exp(-logit)) - target,
    name="Logits BCE Signal",
    short_name="BCE_logits",
    description="Uses sigmoid(logits) - target as blame signal. Maps to BCEWithLogits.",
    legacy_label="BCEWithLogits"
)
SchrodingerSignal = BlameSignal(
    gradient=lambda pred, target: np.ones_like(pred) / _get_n(target),
    name="Schrödinger's Signal",
    short_name="TheCat",
    description="Applies +1 blame no matter what. May confuse your model *and* yourself."
)
