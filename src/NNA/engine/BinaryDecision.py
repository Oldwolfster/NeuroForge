# BinaryDecision.py

class BinaryDecision:
    """Encapsulates binary decision threshold logic and class values."""
    def __init__(self, training_data):
        self.is_active = training_data.is_binary_decision
        if self.is_active:
            self.target_min = training_data.target_min  # e.g., 0
            self.target_max = training_data.target_max  # e.g., 1 (or 100)
            self.label_min = training_data.target_labels[0]  # e.g., "Defaulted"
            self.label_max = training_data.target_labels[1]  # e.g., "Paid It!"
            self.threshold = (self.target_min + self.target_max) / 2

    def decide(self, prediction_unscaled)-> tuple[float, str | None]:
        if not self.is_active:                      return prediction_unscaled, None  # Consistent tuple, label is None for regression
        if prediction_unscaled >= self.threshold:   return self.target_max, self.label_max
        return self.target_min, self.label_min


    def is_true(self, prediction_unscaled, target_unscaled) -> bool:
        """Single source of truth for correctness check."""
        if not self.is_active:
            return None  # Or raise - shouldn't be called for regression
        predicted_class, _ = self.decide(prediction_unscaled)
        return predicted_class == target_unscaled