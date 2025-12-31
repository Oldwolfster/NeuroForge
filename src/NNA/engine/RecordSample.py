from dataclasses import dataclass

@dataclass
class RecordSample:
    run_id: int
    epoch: int
    sample: int
    inputs: str  # Serialized as JSON
    inputs_unscaled: str  # Serialized as JSON
    is_true: int # from BinaryDecision class
    target: float
    target_unscaled: float
    prediction: float  # After threshold(step function) is applied but before unscaling is applied
    prediction_unscaled: float #After unscaling is applied
    prediction_raw: float
    prediction_label: str
    loss_function: str
    loss: float
    loss_gradient: float
    # error: float
    accuracy_threshold: float

    @property
    def error(self):
        return float(self.target - self.prediction_raw)

    @property
    def error_unscaled(self):
        return float(self.target_unscaled - self.prediction_unscaled)

    @property
    def absolute_error_unscaled(self) -> float:
        return float(abs(self.error_unscaled))

    @property
    def absolute_error(self) -> float:
        return float(abs(self.error))

    @property
    def squared_error(self) -> float:
        return self.error ** 2

    @property
    def relative_error(self) -> float:
        return abs(self.error / (self.target + 1e-64))




    @property
    def is_false(self) -> int:
        return not self.is_true

    @property
    def is_true_positive(self) -> int:
        return int(self.is_true and self.target != 0)

    @property
    def is_true_negative(self) -> int:
        return int(self.is_true and self.target == 0)

    @property
    def is_false_positive(self) -> int:
        return int(not self.is_true and self.target == 0)

    @property
    def is_false_negative(self) -> int:
        return int(not self.is_true and self.target != 0)