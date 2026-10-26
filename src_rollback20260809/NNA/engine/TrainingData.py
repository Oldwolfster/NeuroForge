from typing import List, Tuple, Optional, Dict
import math

class TrainingData:
    """
    Manages machine learning training data with support for various normalization methods and metrics.


    """

    def __init__(self,
                 raw_data:      List[Tuple[float, ...]],
                 feature_labels: Optional[List[str]] = None,
                 target_labels:  Optional[List[str]] = None) -> None:
        """
        Initialize the TrainingData instance with raw input data.

        Args:
            raw_data: List of tuples where each tuple contains input features and a target value.
                     The last element of each tuple is treated as the target value.
            feature_labels: List of what each element of the tuple represents
            target_labels:  For binary decision, two classes the targets represent

        Raises:
            ValueError: If data is empty
            ValueError: If data contains inconsistent feature dimensions
            ValueError: If any numeric values are invalid (NaN or infinite)
        """
        self.arena_name          = "Unknown"
        self._raw_data           = None    # initialize private backing field
        self.raw_data            = raw_data  # use setter (it will freeze the input)
        self._fingerprint        = [row[-1] for row in self.raw_data[:10]]

        self._cache              = {}                    # Private dictionary for caching values
        self.feature_labels      = feature_labels        # Optional feature labels
        self.target_labels       = target_labels or ["Beta", "Alpha"]         # Optional outcome labels
        self.is_binary_decision  = self.determine_problem_type()


    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        if self._raw_data is not None:
            raise AttributeError("raw_data is immutable and has already been set.")
        # deep freeze: convert inner sequences to tuples
        self._raw_data = tuple(tuple(row) for row in value)

    def verify_targets_unchanged(self, label=""):
        current = [row[-1] for row in self.raw_data[:10]]
        if current != self._fingerprint:
            print(f"🧨 TARGET CORRUPTION DETECTED during {label}!\nBefore: {self._fingerprint}\nNow:    {current}")
            assert False, "Training data was mutated!"

    def get_list(self):
        """Get the latest version of the data"""
        return self.raw_data

    def get_inputs(self) -> int:
        return [sample[:-1] for sample in self.raw_data]

    def get_targets(self):
        return [sample[-1] for sample in self.raw_data]

    def determine_problem_type(self) -> bool:
        """Examine training data to determine if it's binary decision or regression"""
        unique_values = set(item[-1]    for item in self.raw_data)
        return len(unique_values)  == 2

    @property
    def problem_type(self) -> str:
        if self.is_binary_decision:     return "Binary Decision"
        return "Regression"

    @property
    def input_max(self) -> float:
        if "max_input" not in self._cache:
            if not self.raw_data:
                raise ValueError("Training data is empty; cannot compute max input.")
            self._cache["max_input"] = max(
                max(record[:-1]) for record in self.raw_data
            )
        return self._cache["max_input"]

    @property
    def output_max(self) -> float:
        if "max_output" not in self._cache:
            if not self.raw_data:
                raise ValueError("Training data is empty; cannot compute max output.")
            self._cache["max_output"] = max(t[-1] for t in self.raw_data)
        return self._cache["max_output"]

    @property
    def mean_absolute_target(self) -> float:
        """
        Returns the mean of the absolute target values in the dataset.
        Useful for regression accuracy metrics like 1 - (MAE / mean).
        """
        if "mean_absolute_target" not in self._cache:
            if not self.raw_data:
                raise ValueError("Training data is empty; cannot compute mean target.")
            targets = [abs(t[-1]) for t in self.raw_data]
            self._cache["mean_absolute_target"] = sum(targets) / len(targets)
        return self._cache["mean_absolute_target"]

    def everything_max_magnitude(self) -> float:
        if "everything_max" not in self._cache:
            self._cache["everything_max"] = max(
                abs(item)
                for tup in self.raw_data
                for item in tup
            )
        return self._cache["everything_max"]

    @property
    def sample_count(self) -> int:
        if "sample_count" not in self._cache:
            self._cache["sample_count"] = len(self.raw_data)
        return self._cache["sample_count"]

    @property
    def input_count(self) -> int:
        if "input_count" not in self._cache:
            if not self.raw_data:
                raise ValueError("No data; cannot determine input count.")
            self._cache["input_count"] = len(self.raw_data[0]) - 1
        return self._cache["input_count"]

    def has_outliers(values, threshold_ratio=0.05):
        """
        Detects if more than threshold_ratio of values are outliers based on IQR.
        """
        vs    = sorted(values)
        n     = len(vs)
        if n < 2:
            return False
        def _pct(data, p):
            k = (len(data) - 1) * p / 100
            f = math.floor(k); c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d = k - f
            return data[f] + (data[c] - data[f]) * d
        q1    = _pct(vs, 25)
        q3    = _pct(vs, 75)
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outs  = [v < lower or v > upper for v in values]
        return sum(outs) / n > threshold_ratio

    def is_bounded(values, std_threshold=3):
        """
        Checks if all values lie within mean ± std_threshold·std.
        """
        n     = len(values)
        if n == 0:
            return True
        mean  = sum(values) / n
        var   = sum((v - mean) ** 2 for v in values) / n
        std   = math.sqrt(var)
        lower = mean - std_threshold * std
        upper = mean + std_threshold * std
        return all(lower <= v <= upper for v in values)

    @property
    def target_values(self):
        return [row[-1] for row in self.raw_data]

    @property
    def target_mean(self) -> float:
        if "target_mean" not in self._cache:
            vals = self.target_values
            self._cache["target_mean"] = sum(vals) / len(vals)
        return self._cache["target_mean"]

    @property
    def target_stdev(self) -> float:
        if "target_stdev" not in self._cache:
            vals = self.target_values
            mean = sum(vals) / len(vals)
            var  = sum((v - mean) ** 2 for v in vals) / len(vals)
            self._cache["target_stdev"] = math.sqrt(var)
        return self._cache["target_stdev"]

    @property
    def target_min(self) -> float:
        if "target_min" not in self._cache:
            self._cache["target_min"] = float(min(self.target_values))
        return self._cache["target_min"]

    @property
    def target_max(self) -> float:
        if "target_max" not in self._cache:
            self._cache["target_max"] = float(max(self.target_values))
        return self._cache["target_max"]

    @property
    def target_min_label(self) -> Optional[str]:
        if not self.target_labels or len(set(self.target_values)) != 2:
            return None
        return self.target_labels[0]

    @property
    def target_max_label(self) -> Optional[str]:
        if not self.target_labels or len(set(self.target_values)) != 2:
            return None
        return self.target_labels[1]
